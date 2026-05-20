import numpy as np
import scipy.linalg as la

from .serial_routine import SerialRoutine


class SerialLinalg(SerialRoutine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def trace(self, A, *args, **kwargs):
        r"""Compute the trace of a matrix. This is done via numpy.trace.

        This is the serial version, so it does not use any parallelization. The additional
        keyword arguments are ignored, since they are only relevant for the parallel
        version of the trace routine.

        Parameters
        ----------
            A : np.ndarray
                The input matrix for which to compute the trace.

        Returns
        -------
            float
                The trace of the matrix A.
        """
        return np.trace(A)

    def eigh(self, A, nev, *args, **kwargs):
        r"""Compute eigenvalues and eigenvectors of a matrix. This is done via
        scipy.linalg.eigh.

        This is the serial version, so it does not use any parallelization. The additional
        keyword arguments are ignored, since they are only relevant for the parallel
        version of the eigensolver.

        Parameters
        ----------
            A : np.ndarray
                The input matrix for which to compute eigenvalues and eigenvectors.
            nev : int
                The number of eigenvalues/eigenvectors to compute by ascending order of
                the eigenvalues. If None, all eigenvalues are computed.
        """
        eigenvalues, eigenvectors = la.eigh(
            A, subset_by_index=[0, nev - 1] if nev is not None else None
        )

        return eigenvalues, eigenvectors

    def matmul(
        self,
        A,
        B,
        glob_shape_A=None,
        glob_shape_B=None,
        op_A="N",
        op_B="N",
        slc_idx_A=[[0, None], [0, None]],
        slc_idx_B=[[0, None], [0, None]],
        *args,
        **kwargs,
    ):
        r"""Matrix-matrix multiplication. This is done via numpy.matmul.

        Parameters
        ----------
            A : np.ndarray
                The first matrix to be multiplied.
            B : np.ndarray
                The second matrix to be multiplied.
            glob_shape_A : tuple
                The global shape of matrix A, which is not used in the serial version but
                is relevant for the parallel version.
            glob_shape_B : tuple
                The global shape of matrix B, which is not used in the serial version but
                is relevant for the parallel version.
            op_A : str, optional
                The operation to apply to matrix A before multiplication. It can be 'N'
                for no operation, 'T' for transpose, or 'C' for conjugate transpose.
                Default is 'N'.
            op_B : str, optional
                The operation to apply to matrix B before multiplication. It can be 'N'
                for no operation, 'T' for transpose, or 'C' for conjugate transpose.
                Default is 'N'.
            slc_idx_A : list of lists, optional
                The slice indices for matrix A. Default is [[0, None], [0, None]], which
                means no slicing.
            slc_idx_B : list of lists, optional
                The slice indices for matrix B. Default is [[0, None], [0, None]], which
                means no slicing.

        Returns
        -------
            np.ndarray
                The result of the matrix multiplication A @ B.
        """
        IA, IA_end = slc_idx_A[0]  # Row slicing indices
        JA, JA_end = slc_idx_A[1]  # Col slicing indices
        IB, IB_end = slc_idx_B[0]  # Row slicing indices
        JB, JB_end = slc_idx_B[1]  # Col slicing indices

        if IA_end is None:
            IA_end = A.shape[0]
        if JA_end is None:
            JA_end = A.shape[1]
        if IB_end is None:
            IB_end = B.shape[0]
        if JB_end is None:
            JB_end = B.shape[1]
        A_sliced = A[IA:IA_end, JA:JA_end]
        B_sliced = B[IB:IB_end, JB:JB_end]

        if op_A == "T":
            A_sliced = A_sliced.T
        elif op_A == "C":
            A_sliced = A_sliced.conj().T
        if op_B == "T":
            B_sliced = B_sliced.T
        elif op_B == "C":
            B_sliced = B_sliced.conj().T
        return A_sliced @ B_sliced

    def commutator(self, loc_A, loc_B, *args, **kwargs) -> np.ndarray:
        r"""Compute the commutator :math:`[A,B] = AB - BA` for matrices :math:`A` and
        :math:`B`.

        This is the serial version, so it does not use any parallelization.

        Parameters
        ----------
            loc_A : np.ndarray
                The first matrix in the commutator.
            loc_B : np.ndarray
                The second matrix in the commutator.

        Returns
        -------
            np.ndarray
                The commutator of A and B, given by AB - BA.
        """
        return loc_A @ loc_B - loc_B @ loc_A

    def linsolve(
        self,
        A,
        B,
        glob_shape_A,
        glob_shape_B,
        slc_idx_A=[[0, None], [0, None]],
        slc_idx_B=[[0, None], [0, None]],
        *args,
        **kwargs,
    ) -> np.ndarray:
        r"""Solve the linear problem :math:`Ax=B` for :math:`x`. Here, A is a matrix
        and B is a vector or matrix of right-hand sides.

        Parameters
        ----------
            A : np.ndarray
                The coefficient matrix A in the linear system.
            B : np.ndarray
                The right-hand side vector or matrix B in the linear system.
            glob_shape_A : tuple
                The global shape of matrix A, which is not used in the serial version but
                is relevant for the parallel version.
            glob_shape_B : tuple
                The global shape of matrix B, which is not used in the serial version but
                is relevant for the parallel version.
            slc_idx_A : list of lists, optional
                The slice indices for matrix A. Default is [[0, None], [0, None]], which
                means no slicing.
            slc_idx_B : list of lists, optional
                The slice indices for matrix B. Default is [[0, None], [0, None]], which
                means no slicing.

        Returns
        -------
            np.ndarray
                The solution X to the linear system AX = B.
        """
        IA, IA_end = slc_idx_A[0]  # Row slicing indices
        JA, JA_end = slc_idx_A[1]  # Col slicing indices
        IB, IB_end = slc_idx_B[0]  # Row slicing indices
        JB, JB_end = slc_idx_B[1]  # Col slicing indices

        if IA_end is None:
            IA_end = A.shape[0]
        if JA_end is None:
            JA_end = A.shape[1]
        if IB_end is None:
            IB_end = B.shape[0]
        if JB_end is None:
            JB_end = B.shape[1]
        A_sliced = A[IA:IA_end, JA:JA_end]
        B_sliced = B[IB:IB_end, JB:JB_end]

        return la.solve(A_sliced, B_sliced)
