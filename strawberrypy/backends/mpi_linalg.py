import numpy as np
from mpi4py import MPI

from .mpi_routine import MPIRoutine


class MPILinalg(MPIRoutine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def trace(self, loc_mat, N, root=None, dtype=None):
        r"""Compute the trace of a matrix. The matrix is assumed to be
        distributed in block-cyclic layout. Each rank sums its local diagonal
        entries, followed by an MPI reduce.

        Parameters
        ----------
            loc_mat : np.ndarray
                Local block of the distributed matrix.
            N : int
                Global matrix size (assuming square).
            root : int or None
                Rank to gather the trace to. If None, an Allreduce is performed
                and all ranks receive the trace.
            dtype : data-type or None
                Dtype of the elements. If None, inferred from `loc_mat`.

        Returns
        -------
            Scalar value or None
                The trace of the global matrix on `root` (or all ranks if
                `root` is None). None on non-root ranks if `root` is not None.
        """
        if dtype is None:
            dtype = loc_mat.dtype

        np_dt = np.dtype(dtype)
        k_global = np.arange(N)
        owner_rows = (k_global // self.nblk) % self.nprow
        owner_cols = (k_global // self.nblk) % self.npcol
        am_i_owner = (owner_rows == self.myprow) & (owner_cols == self.mypcol)
        my_global_indices = k_global[am_i_owner]

        local_trace = np.zeros(1, dtype=np_dt)
        if my_global_indices.size > 0:
            n_cycles_row = (my_global_indices // self.nblk) // self.nprow
            n_cycles_col = (my_global_indices // self.nblk) // self.npcol
            block_offset = my_global_indices % self.nblk
            local_rows = n_cycles_row * self.nblk + block_offset
            local_cols = n_cycles_col * self.nblk + block_offset
            local_trace[0] = np.sum(loc_mat[local_rows, local_cols])

        global_trace = np.zeros(1, dtype=np_dt)
        mpi_op_type = MPI._typedict[np_dt.char]

        if root is None:
            self.comm.Allreduce(
                [local_trace, mpi_op_type], [global_trace, mpi_op_type], op=MPI.SUM
            )
            return global_trace[0]
        else:
            self.comm.Reduce(
                [local_trace, mpi_op_type],
                [global_trace, mpi_op_type],
                op=MPI.SUM,
                root=root,
            )
            if self.mpi_rank == root:
                return global_trace[0]
            return None

    def eigh(
        self, A, N=None, nev=None, collect_evec=False, debug_info=True
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Compute eigenvalues and eigenvectors of a matrix.
        Matrix A is distributed in block-cyclic layout with block size nblk.

        Parameters
        ----------
            A : np.ndarray
                Local block of the distributed matrix (Fortran-ordered).
            N : int
                Global matrix size (assuming square).
            nev : int
                Number of eigenvalues/eigenvectors to compute by ascending order of
                eigenvalues. If None, all eigenvalues are computed.
            collect_evec : bool
                Whether to collect and return eigenvectors (default: False).
            debug_info : bool
                Whether to print debug information about the distribution and gathering
                (default: True).

        Returns
        -------
            eigenvalues : np.ndarray
                Array of computed eigenvalues (length N).
            eigenvectors : np.ndarray
                If collect_evec is True, the global eigenvector matrix (N x N) on root
                rank and None on other ranks. If collect_evec is False, the local
                eigenvector block on each rank (Fortran-ordered). For columns > nev, the
                values are set to zero by the eigensolver.
        """
        from .fortran.diagonalize import diagonalize

        if nev is None:
            nev = N

        if self.is_master_rank and debug_info:
            print(f"Using eigensolver", flush=True)
            print(f"Matrix size {N} x {N} with block size {self.nblk}", flush=True)
            print(f"Running on {self.mpi_size} MPI ranks", flush=True)

        info = (
            f"Rank: {self.mpi_rank} has {A.shape[0]} rows and {A.shape[1]} columns. "
            + f"{A.shape[1] * A.shape[0] / N**2 * 100:.2f}% of the global matrix."
        )
        all_info = self.comm.gather(info, root=0)
        if self.is_master_rank and debug_info:
            for line in all_info:
                print(line, flush=True)

        local_mat = np.empty(np.shape(A), dtype=np.complex128, order="F")
        local_mat[:] = A

        # ========= Call the wrapper ========= #
        eigenvalues = np.empty(N, dtype=np.float64)
        local_evec = np.empty(np.shape(A), dtype=np.complex128, order="F")
        info = diagonalize(
            self.fcomm,
            self.nblk,
            nev,
            self.nprow,
            self.npcol,
            local_mat[:],
            eigenvalues[:],
            local_evec[:],
        )

        if info != 0:
            raise RuntimeError(f"Diagonalization failed with info={info}")
        if self.is_master_rank and debug_info:
            if N >= 5:
                print(f"eigenvalues (first 5 lowest): {eigenvalues[:5]}", flush=True)
            else:
                print(f"eigenvalues (first {N} lowest): {eigenvalues[:5]}", flush=True)

        if not self.is_master_rank:
            eigenvalues = np.empty(
                0, dtype=np.float64
            )  # Non-root ranks return empty eigenvalues
        if collect_evec:

            global_evec = self.gather(local_evec, N, N, root=0, dtype=np.complex128)

            return eigenvalues, global_evec
        return eigenvalues, local_evec

    def matmul(
        self,
        loc_A,
        loc_B,
        glob_shape_A,
        glob_shape_B,
        op_A="N",
        op_B="N",
        slc_idx_A=[[0, None], [0, None]],
        slc_idx_B=[[0, None], [0, None]],
        get_shape=False,
        dtype=np.complex128,
    ) -> np.ndarray | tuple[np.ndarray, tuple[int, int]]:
        r"""Matrix-matrix multiplication. A and B are distributed matrices:

            C(M,N) = alpha * op(A)[M,K] * op(B)[K,N] + beta * C(M,N)

        The shape of the resulting global matrix is (M,N), K is the inner dimension and
        alpha, beta are scalar constants.

        Parameters
        ----------
            loc_A and loc_B:
                Submatrices of A and B local to each processor.
            glob_shape_A and  glob_shape_B:
                Shape (row,column) of the global matrices A and B, respectively, without
                any operations or slicing.
            op_A and op_B:
                Operation on matrices A and B. Operation will be done after applying
                slicing. It can be 'N'-> do nothing, 'T'-> transpose, 'C'-> conjugate
                transpose.
            slc_idx_A:
                Indices used when slicing global matrix A. For example: A[0:,7:12] => slc_idx_A = [[0,None], [7,12]]
            slc_idx_B:
                Indices used when slicing global matrix B. For example: B[:4,5:] => slc_idx_B = [[0,4], [5,None]]
            get_shape:
                If python:`True`, return also the shape of the resulting matrix.
            dtype:
                Datatype of the matrix.

        Returns
        -------
            loc_C:
                Submatrix of the resulting matrix C local to each processor. The global
                shape of C can be inferred from the input parameters.
        """
        from .fortran.matmul import matmul

        IA, IA_end = slc_idx_A[0]  # Row slicing indices
        JA, JA_end = slc_idx_A[1]  # Col slicing indices
        IB, IB_end = slc_idx_B[0]  # Row slicing indices
        JB, JB_end = slc_idx_B[1]  # Col slicing indices

        if IA_end is None:
            IA_end = glob_shape_A[0]
        if JA_end is None:
            JA_end = glob_shape_A[1]
        if IB_end is None:
            IB_end = glob_shape_B[0]
        if JB_end is None:
            JB_end = glob_shape_B[1]

        # Using python 0-based index
        sliced_shape_A = [IA_end - IA, JA_end - JA]
        sliced_shape_B = [IB_end - IB, JB_end - JB]

        if op_A.upper() == "N" and op_B.upper() == "N":
            M, N = sliced_shape_A[0], sliced_shape_B[1]
            K = sliced_shape_A[1]
        elif op_A.upper() in ["T", "C"] and op_B.upper() == "N":
            M, N = sliced_shape_A[1], sliced_shape_B[1]
            K = sliced_shape_A[0]
        elif op_A.upper() == "N" and op_B.upper() in ["T", "C"]:
            M, N = sliced_shape_A[0], sliced_shape_B[0]
            K = sliced_shape_A[1]
        else:
            M, N = sliced_shape_A[1], sliced_shape_B[0]
            K = sliced_shape_A[0]

        # Determine local matrix sizes and create local arrays
        lr_C = self.numroc(M, self.nblk, self.myprow, 0, self.nprow)
        lc_C = self.numroc(N, self.nblk, self.mypcol, 0, self.npcol)

        loc_C = np.empty((lr_C, lc_C), dtype=dtype, order="F")

        # Call the Fortran wrapper - loc_C array will be modified in place
        alpha = 1.0 + 0.0j
        beta = 0.0 + 0.0j

        # Fmt: off
        info = matmul(
            M,
            K,
            N,
            self.nblk,
            self.nprow,
            self.npcol,
            alpha,
            beta,
            loc_A,
            loc_B,
            loc_C,
            glob_shape_A[0],
            glob_shape_A[1],
            glob_shape_B[0],
            glob_shape_B[1],
            op_A,
            op_B,
            IA + 1,
            JA + 1,
            IB + 1,
            JB + 1,  # Using fortran 1-based index
        )
        # Fmt: on
        if info != 0:
            raise RuntimeError(f"Matrix multiplication failed with info={info}")

        if get_shape:
            return loc_C, (M, N)

        return loc_C

    def commutator(self, loc_A, loc_B, glob_shape_A, glob_shape_B) -> np.ndarray:
        r"""Compute the commutator :math:`[A,B] = AB - BA` for matrices :math:`A` and
        :math:`B`. The matrices are distributed in block-cyclic layout.

        Parameters
        ----------
            loc_A and loc_B:
                Submatrices of A and B local to each processor.
            glob_shape_A and glob_shape_B:
                Global shapes of matrices A and B.
        Returns
        -------
            loc_comm:
                Submatrix of the resulting commutator [A,B] local to each processor.
        """
        vals = self.matmul(loc_A, loc_B, glob_shape_A, glob_shape_B) - self.matmul(
            loc_B, loc_A, glob_shape_B, glob_shape_A
        )

        return vals

    def linsolve(
        self,
        loc_A_arr,
        loc_B_arr,
        glob_shape_A,
        glob_shape_B,
        slc_idx_A=[[0, None], [0, None]],
        slc_idx_B=[[0, None], [0, None]],
    ):
        r"""
        Solve the linear problem :math:`Ax=B` for :math:`x`.
        Assumes matrices are already distributed.

        Parameters
        ----------
            loc_A_arr :
                Local matrix containing the coefficient square matrix A
            loc_B_arr :
                Local matrix containing matrix B
            glob_shape_A, glob_shape_B :
                Shape (row, column) of the global matrices A and B, respectively,
                without any operations or slicing.
            slc_idx_A:
                Indices used when slicing global matrix A. For example: A[0:,7:12] => slc_idx_A = [[0,None],[7,12]]
            slc_idx_B:
                Indices used when slicing global matrix B. For example: B[:4,5:] =>  slc_idx_B = [[0,4],[5,None]]
        """
        from .fortran.linsolve import linsolve

        gr_A, gc_A = glob_shape_A
        gr_B, gc_B = glob_shape_B

        # Using python 0-based index
        IA, IA_end = slc_idx_A[0]  # Row slicing indices
        JA, JA_end = slc_idx_A[1]  # Col slicing indices
        IB, IB_end = slc_idx_B[0]  # Row slicing indices
        JB, JB_end = slc_idx_B[1]  # Col slicing indices

        if IA_end is None:
            IA_end = glob_shape_A[0]
        if JA_end is None:
            JA_end = glob_shape_A[1]
        if IB_end is None:
            IB_end = glob_shape_B[0]
        if JB_end is None:
            JB_end = glob_shape_B[1]

        N = IA_end - IA
        nrhs = JB_end - JB

        #  Call the Fortran ScaLAPACK Wrapper
        info = linsolve(
            N,
            nrhs,
            self.nblk,
            loc_A_arr,
            gr_A,
            loc_B_arr,
            gr_B,
            gc_B,
            self.nprow,
            self.npcol,
            IA + 1,
            JA + 1,
            IB + 1,
            JB + 1,
        )  # Using fortran 1-based index

        if info != 0:
            raise RuntimeError(f"Linear solve failed with info={info}")

        return loc_B_arr
