import numpy as np
from mpi4py import MPI


class MPIRoutine:
    def __init__(self, nblk):
        self.comm = MPI.COMM_WORLD
        self.MPI = MPI
        self.mpi_rank = self.comm.Get_rank()
        self.mpi_size = self.comm.Get_size()
        self.fcomm = self.comm.py2f()  # Get Fortran communicator handle
        self.is_master_rank = self.mpi_rank == 0

        # Determine process grid dimensions. Processors distributed by row first, so nprow
        #   (npcol) is number of rows (cols). I choose nprow and npcol to be as close to
        #   each other as possible to minimize communication overhead.
        self.nprow = int(np.floor(np.sqrt(self.mpi_size)))
        while self.mpi_size % self.nprow != 0:
            self.nprow -= 1
        self.npcol = self.mpi_size // self.nprow

        # Determine my position in the BLACS grid
        self.myprow = self.mpi_rank // self.npcol
        self.mypcol = self.mpi_rank % self.npcol

        # Set block size
        self.nblk = nblk

    def numroc(self, N, NB, IPROC, ISRCPROC, NPROCS) -> int:
        r"""Calculates the number of rows or columns a process owns.
        This is a Python translation of the ScaLAPACK numroc tool.

        Parameters
        ----------
            N: int
                Total number of rows/columns.
            NB: int
                Block size.
            IPROC: int
                Rank of the process.
            ISRCPROC: int
                Rank of the source process.
            NPROCS: int
                Total number of processes.
        """
        # Calculate distance from source process
        proc_dist = (NPROCS + IPROC - ISRCPROC) % NPROCS

        # Number of full blocks
        num_blocks = (N - 1) // NB

        # My number of blocks
        my_blocks = num_blocks // NPROCS
        if proc_dist < (num_blocks % NPROCS):
            my_blocks += 1

        # My total number of elements
        n_elems = my_blocks * NB

        # Add remainder if I own the last block
        if proc_dist == (num_blocks % NPROCS):
            n_elems += (N - 1) % NB + 1

        return n_elems

    def compute_indices(self, size, nprocs, proc_rank, block_size) -> tuple[list, list]:
        r"""Given a global matrix size (M, N), process grid dimensions (nprow,
        npcol), process coordinates (prow, pcol), and block size, return the
        list of global row (column) indices and also the corresponding local
        indices of the submatrix contained by a process under block cyclic
        distribution.

        Size can be M (N), nprocs can be nprow (npcol), proc_rank can be
        prow (pcol).
        """
        g_indices = []
        num_full_blocks = size // block_size
        remaining = size % block_size

        for block_idx in range(num_full_blocks):
            # Each full block
            for b in range(block_size):
                global_idx = block_idx * nprocs * block_size + proc_rank * block_size + b
                if global_idx < size:
                    g_indices.append(global_idx)

        # Handle remaining part if size is not divisible
        offset = num_full_blocks * nprocs * block_size
        for b in range(remaining):
            global_idx = offset + proc_rank * block_size + b
            if global_idx < size:
                g_indices.append(global_idx)

        # Calculate local indices
        l_indices = [
            (i // (block_size * nprocs)) * block_size + (i % block_size)
            for i in g_indices
        ]

        return l_indices, g_indices

    def distribute(self, glob_A, root=0, dtype=None) -> np.ndarray:
        r"""Distribute a global matrix `glob_A` defined on `root` rank to all
        processes using a ScaLAPACK-style block-cyclic layout. This implementation
        uses per-rank `Send`/`Recv` of contiguous Fortran-ordered local blocks to avoid
        pickling overhead.

        Parameters
        ----------
            glob_A : np.ndarray or None
                Global matrix on `root` rank, None on other ranks.
            root : int
                Root rank holding the global matrix.
            dtype : numpy dtype
                Dtype for local arrays (default: complex128).

        Returns
        -------
            loc_A : np.ndarray
                Local subarray owned by this process (Fortran-ordered).
        """
        if self.mpi_rank == root:
            if glob_A is None:
                raise ValueError("Glob_A must be provided on root rank")
            M, N = np.shape(glob_A)
            glob_A = np.asarray(glob_A, dtype=dtype, order="F")
        else:
            M, N = None, None

        M, N = self.comm.bcast((M, N), root=root)

        if dtype is None:
            dtype = self.comm.bcast(
                glob_A.dtype if self.mpi_rank == root else None, root=root
            )

        # Map dtype to MPI datatype when possible
        np_dt = np.dtype(dtype)
        mpi_type = MPI._typedict[np_dt.char]

        def _build_local_block(proc_row, proc_col):
            lr_A = self.numroc(M, self.nblk, proc_row, 0, self.nprow)
            lc_A = self.numroc(N, self.nblk, proc_col, 0, self.npcol)

            loc_proc = np.empty((lr_A, lc_A), dtype=dtype, order="F")
            lri, gri = self.compute_indices(M, self.nprow, proc_row, self.nblk)
            lci, gci = self.compute_indices(N, self.npcol, proc_col, self.nblk)
            loc_proc[np.ix_(lri, lci)] = glob_A[np.ix_(gri, gci)]
            return loc_proc

        # Root streams one rank block at a time to avoid payload accumulation
        if self.mpi_rank == root:
            for proc in range(self.mpi_size):
                proc_row = proc // self.npcol
                proc_col = proc % self.npcol
                loc_proc = _build_local_block(proc_row, proc_col)
                if proc == root:
                    loc_A = loc_proc
                elif mpi_type is not None:
                    self.comm.Send([loc_proc, mpi_type], dest=proc, tag=77)
                else:
                    self.comm.send(loc_proc, dest=proc, tag=77)

        else:
            # Non-root ranks allocate local array then receive
            lr_A = self.numroc(M, self.nblk, self.myprow, 0, self.nprow)
            lc_A = self.numroc(N, self.nblk, self.mypcol, 0, self.npcol)
            loc_A = np.empty((lr_A, lc_A), dtype=dtype, order="F")
            if mpi_type is not None:
                self.comm.Recv([loc_A, mpi_type], source=root, tag=77)
            else:
                loc_A = self.comm.recv(source=root, tag=77)

        return loc_A

    def distribute_diag(
        self, diag_values, root=0, dtype=None, *args, **kwargs
    ) -> np.ndarray:
        r"""Distribute a 1D array ``diag_values`` (defined on ``root`` rank)
        onto the diagonal of a global ``N x N`` matrix stored in block-cyclic
        layout. Only local blocks are created on each rank; off-diagonal local
        entries are zero.

        Parameters
        ----------
            diag_values : np.ndarray | None
                Diagonal values on ``root`` rank. Non-root ranks can pass
                :python:`None`.
            root : int
                Rank holding ``diag_values``.
            dtype : data-type | None
                Dtype of the distributed matrix. If :python:`None`, inferred from
                ``diag_values`` on ``root``.

        Returns
        -------
            loc_A : np.ndarray
                Local block of the distributed matrix with only diagonal entries
                set.
        """
        if self.mpi_rank == root:
            if diag_values is None:
                raise ValueError("Diag_values must be provided on root rank")

            if diag_values.ndim != 1:
                raise ValueError("Diag_values must be a 1D array")
            N = diag_values.size

        else:
            N = None

        N = self.comm.bcast(N, root=root)

        if dtype is None:
            dtype = self.comm.bcast(
                diag_values.dtype if self.mpi_rank == root else None, root=root
            )

        # Map dtype to MPI datatype when possible
        np_dt = np.dtype(dtype)
        mpi_type = MPI._typedict[np_dt.char]

        lr_A = self.numroc(N, self.nblk, self.myprow, 0, self.nprow)
        lc_A = self.numroc(N, self.nblk, self.mypcol, 0, self.npcol)
        loc_A = np.zeros((lr_A, lc_A), dtype=np_dt, order="F")

        tag_meta = 103
        tag_rows = 104
        tag_cols = 105
        tag_vals = 106
        if self.mpi_rank == root:
            for proc in range(self.mpi_size):
                proc_row = proc // self.npcol
                proc_col = proc % self.npcol

                lri, gri = self.compute_indices(N, self.nprow, proc_row, self.nblk)
                lci, gci = self.compute_indices(N, self.npcol, proc_col, self.nblk)

                gri_arr = np.asarray(gri, dtype=np.int64)
                gci_arr = np.asarray(gci, dtype=np.int64)
                lri_arr = np.asarray(lri, dtype=np.int64)
                lci_arr = np.asarray(lci, dtype=np.int64)

                common_g, ridx, cidx = np.intersect1d(
                    gri_arr, gci_arr, return_indices=True
                )

                if common_g.size == 0:
                    rows = np.empty(0, dtype=np.int64)
                    cols = np.empty(0, dtype=np.int64)
                    vals = np.empty(0, dtype=np_dt)
                else:
                    rows = lri_arr[ridx]
                    cols = lci_arr[cidx]
                    vals = diag_values[common_g]

                if proc == root:
                    if vals.size:
                        loc_A[rows, cols] = vals
                else:
                    count = np.array([rows.size], dtype=np.int64)
                    self.comm.Send([count, MPI.LONG_LONG], dest=proc, tag=tag_meta)
                    if rows.size > 0:
                        self.comm.Send([rows, MPI.LONG_LONG], dest=proc, tag=tag_rows)
                        self.comm.Send([cols, MPI.LONG_LONG], dest=proc, tag=tag_cols)
                        if mpi_type is not None:
                            self.comm.Send([vals, mpi_type], dest=proc, tag=tag_vals)
                        else:
                            vals_bytes = np.ascontiguousarray(vals).view(np.uint8)
                            self.comm.Send(
                                [vals_bytes, MPI.BYTE], dest=proc, tag=tag_vals
                            )
        else:
            count = np.empty(1, dtype=np.int64)
            self.comm.Recv([count, MPI.LONG_LONG], source=root, tag=tag_meta)
            n_diag = int(count[0])
            if n_diag > 0:
                rows = np.empty(n_diag, dtype=np.int64)
                cols = np.empty(n_diag, dtype=np.int64)
                vals = np.empty(n_diag, dtype=np_dt)
                self.comm.Recv([rows, MPI.LONG_LONG], source=root, tag=tag_rows)
                self.comm.Recv([cols, MPI.LONG_LONG], source=root, tag=tag_cols)
                if mpi_type is not None:
                    self.comm.Recv([vals, mpi_type], source=root, tag=tag_vals)
                else:
                    vals_bytes = vals.view(np.uint8)
                    self.comm.Recv([vals_bytes, MPI.BYTE], source=root, tag=tag_vals)
                loc_A[rows, cols] = vals

        return loc_A

    def gather(self, loc_A, M, N, root=0, dtype=None) -> np.ndarray:
        r"""Gather local block-cyclic pieces from all ranks and reconstruct the
        global matrix on `root` in a memory-efficient way (one block at a time).

        Parameters
        ----------
            loc_A : np.ndarray
                Local block owned by this rank (Fortran-ordered).
            M, N : int
                Global matrix shape.
            root : int
                Rank that will receive and reconstruct the global matrix.
            dtype : numpy dtype
                Dtype of the global matrix.

        Returns
        -------
            glob_A : np.ndarray or None
                Reconstructed global matrix on `root`, None on other ranks.
        """
        if dtype is None:
            dtype = self.comm.bcast(
                loc_A.dtype if self.mpi_rank == root else None, root=root
            )

        # Map dtype to MPI datatype when possible
        np_dt = np.dtype(dtype)
        mpi_type = MPI._typedict[np_dt.char]

        if self.mpi_rank == root:
            glob_A = np.empty((M, N), dtype=dtype, order="F")
            for proc in range(self.mpi_size):
                proc_row = proc // self.npcol
                proc_col = proc % self.npcol

                lr = self.numroc(M, self.nblk, proc_row, 0, self.nprow)
                lc = self.numroc(N, self.nblk, proc_col, 0, self.npcol)

                lri, gri = self.compute_indices(M, self.nprow, proc_row, self.nblk)
                lci, gci = self.compute_indices(N, self.npcol, proc_col, self.nblk)

                if proc == root:
                    buf = loc_A
                else:
                    buf = np.empty((lr, lc), dtype=dtype, order="F")
                    if mpi_type is not None:
                        self.comm.Recv([buf, mpi_type], source=proc, tag=88)
                    else:
                        buf = self.comm.recv(source=proc, tag=88)

                # Place into global matrix
                glob_A[np.ix_(gri, gci)] = buf[np.ix_(lri, lci)]

            return glob_A
        else:
            # Non-root ranks send their local block
            if mpi_type is not None:
                self.comm.Send([loc_A, mpi_type], dest=root, tag=88)
            else:
                self.comm.send(loc_A, dest=root, tag=88)
            return None

    def get_diag(self, loc_mat, N, root=0, dtype=None, broadcast=False) -> np.ndarray:
        r"""Extract diagonal elements of a global NxN matrix that is stored
        distributed in block-cyclic layout. Each rank provides its local block
        `loc_mat` (Fortran-ordered). If `root` is specified, the full diagonal
        (length N) is reconstructed on `root` and returned; other ranks receive
        `None`. If `root` is None, each rank returns its list of (global_index,
        value) pairs.

        This method streams contributions to `root` to avoid
        accumulating large intermediate structures.

        Parameters
        ----------
            loc_mat : np.ndarray
                Local block of the distributed matrix (Fortran-ordered).
            N : int
                Global matrix size (assuming square).
            root : int or None
                Rank to gather the full diagonal. If None, no gathering is done and
                each rank returns its local diagonal contributions as (global_index,
                value) pairs.
            dtype : data-type or None
                Dtype of the diagonal values. If None, inferred from `loc_mat`.
            broadcast : bool
                If True and `root` is not None, broadcast the gathered diagonal from
                `root` to all ranks after gathering.

        Returns
        -------
            np.ndarray or list of (int, value) pairs
                If `root` is not None, the full diagonal array on `root` and None on
                other ranks (or the full diagonal broadcast to all ranks if
                `broadcast` is True). If `root` is None, a list of (global_index,
                value) pairs corresponding to the diagonal entries owned by this rank.
        """
        if dtype is None:
            dtype = loc_mat.dtype

        # Compute which global diagonal indices this rank owns
        k_global = np.arange(N)
        owner_rows = (k_global // self.nblk) % self.nprow
        owner_cols = (k_global // self.nblk) % self.npcol
        am_i_owner = (owner_rows == self.mpi_rank // self.npcol) & (
            owner_cols == self.mpi_rank % self.npcol
        )
        my_global_indices = k_global[am_i_owner]

        if my_global_indices.size == 0:
            my_pairs = []
        else:
            # Compute local positions inside loc_mat corresponding to diagonal
            n_cycles_row = (my_global_indices // self.nblk) // self.nprow
            n_cycles_col = (my_global_indices // self.nblk) // self.npcol
            block_offset = my_global_indices % self.nblk
            local_rows = n_cycles_row * self.nblk + block_offset
            local_cols = n_cycles_col * self.nblk + block_offset
            my_values = loc_mat[local_rows, local_cols]
            my_pairs = list(zip(my_global_indices.tolist(), my_values.tolist()))

        tag_meta = 199
        tag_idx = 200
        tag_vals = 201

        if root is None:
            return my_pairs

        # Convert pairs to separate arrays for zero-copy transfer
        if len(my_pairs) == 0:
            my_idxs = np.empty(0, dtype=np.int64)
            my_vals = np.empty(0, dtype=dtype)
        else:
            my_idxs = np.asarray([p[0] for p in my_pairs], dtype=np.int64)
            my_vals = np.asarray([p[1] for p in my_pairs], dtype=dtype)

        # Map dtype to MPI datatype when possible
        np_dt = np.dtype(dtype)
        mpi_type = MPI._typedict[np_dt.char]

        if self.mpi_rank == root:
            diag = np.empty(N, dtype=dtype)
            # Place own
            if my_idxs.size:
                diag[my_idxs] = my_vals

            # Receive from others using low-level buffers
            for proc in range(self.mpi_size):
                if proc == root:
                    continue

                # Receive count metadata (one int64)
                count = np.empty(1, dtype=np.int64)
                self.comm.Recv([count, MPI.LONG_LONG], source=proc, tag=tag_meta)
                n = int(count[0])
                if n == 0:
                    continue

                # Receive indices
                idx_buf = np.empty(n, dtype=np.int64)
                self.comm.Recv([idx_buf, MPI.LONG_LONG], source=proc, tag=tag_idx)

                # Receive values
                if mpi_type is not None:
                    val_buf = np.empty(n, dtype=np_dt)
                    self.comm.Recv([val_buf, mpi_type], source=proc, tag=tag_vals)
                else:
                    val_buf = np.empty(n, dtype=np_dt)
                    val_bytes = val_buf.view(np.uint8)
                    self.comm.Recv([val_bytes, MPI.BYTE], source=proc, tag=tag_vals)

                diag[idx_buf] = val_buf

            if broadcast:
                # Broadcast in-place using low-level buffer
                # Root broadcasts a contiguous array `diag`
                self.comm.Bcast([diag, MPI._typedict[np_dt.char]], root=root)
                return diag

            return diag
        else:
            # Send count metadata
            count = np.array([my_idxs.size], dtype=np.int64)
            self.comm.Send([count, MPI.LONG_LONG], dest=root, tag=tag_meta)
            if my_idxs.size > 0:
                # Send indices
                self.comm.Send([my_idxs, MPI.LONG_LONG], dest=root, tag=tag_idx)

                # Send values using appropriate MPI type or bytes
                if mpi_type is not None:
                    self.comm.Send([my_vals, mpi_type], dest=root, tag=tag_vals)
                else:
                    my_vals_bytes = np.ascontiguousarray(my_vals).view(np.uint8)
                    self.comm.Send([my_vals_bytes, MPI.BYTE], dest=root, tag=tag_vals)

            if broadcast:
                # Participate in broadcast to receive diag
                # Prepare receive buffer
                diag = np.empty(N, dtype=dtype)
                self.comm.Bcast([diag, MPI._typedict[np_dt.char]], root=root)
                return diag

            return np.empty(0, dtype=dtype)

    def set_zero_off_diag(self, loc_mat, N) -> np.ndarray:
        r"""Given a local block `loc_mat` of a global NxN matrix distributed in
        block-cyclic layout, set all off-diagonal entries to zero while keeping
        diagonal entries intact. This is a local operation that does not require
        communication.

        Parameters
        ----------
            loc_mat : np.ndarray
                Local block of the distributed matrix (Fortran-ordered).
            N : int
                Global matrix size (assuming square).
        Returns
        -------
            np.ndarray
                Local block of the distributed matrix with off-diagonal entries set to zero.
        """
        _, gri = self.compute_indices(N, self.nprow, self.myprow, self.nblk)
        _, gci = self.compute_indices(N, self.npcol, self.mypcol, self.nblk)

        # Create boolean mask of local positions that are on the global diagonal
        row_globals = np.array(gri)[:, None]
        col_globals = np.array(gci)[None, :]
        diag_mask = row_globals == col_globals

        # Zero all entries that are NOT on the global diagonal
        loc_mat[~diag_mask] = 0.0

        return loc_mat

    def shared_array(
        self, root_array: np.ndarray, root: int = 0, dtype=None, get_win=False
    ) -> np.ndarray:
        r"""Create a node-local shared array from data owned by the root rank.

        The data flow is:
        1. Split ``comm`` with ``MPI.COMM_TYPE_SHARED`` so each node gets its own
        shared-memory communicator.
        2. Keep only the node leaders in a second communicator.
        3. Broadcast the root-owned payload to the leaders.
        4. Let each leader publish that payload into a shared window for the local
        node.
        5. Return a NumPy view on the shared buffer.

        Parameters
        ----------
        root_array:
            Array owned by ``root``. Other ranks may pass ``None``.
        root:
            World rank that owns the initial array.
        dtype:
            Data type of the array. Required if ``root_array`` is not provided on this rank.
        get_win:
            Whether to return the MPI window object along with the shared array.

        Returns
        -------
        shared_array, win, node_comm, leader_comm
            ``shared_array`` is the NumPy view backed by the shared window,
            ``win`` is the shared-memory window, ``node_comm`` is the node-local
            communicator, and ``leader_comm`` is the communicator of node leaders.

        Notes
        -----
        This helper assumes that ``root`` is also a node leader in its shared-memory
        communicator. That is the standard layout when the root rank is rank 0.
        """
        # Create shared-memory communicator for each node
        node_comm = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
        node_rank = node_comm.Get_rank()

        # Create communicator of node leaders
        leader_comm = self.comm.Split(color=(node_rank == 0), key=self.mpi_rank)

        if dtype is None:
            dtype = self.comm.bcast(
                root_array.dtype if self.mpi_rank == root else None, root=root
            )

        # Map dtype to MPI datatype when possible
        np_dt = np.dtype(dtype)
        mpi_type = MPI._typedict[np_dt.char]

        if self.mpi_rank == root:
            M, N = (root_array.shape[0], 1) if root_array.ndim == 1 else root_array.shape
        else:
            M, N = None, None

        M = self.comm.bcast(M, root=root)
        N = self.comm.bcast(N, root=root)

        nbytes = M * N * np.dtype(dtype).itemsize if node_rank == 0 else 0
        win = MPI.Win.Allocate_shared(nbytes, np.dtype(dtype).itemsize, comm=node_comm)
        buf, _ = win.Shared_query(0)

        shape = (M, N) if N > 1 else (M,)
        shared_A = np.ndarray(buffer=buf, dtype=dtype, shape=shape)

        # Map dtype to MPI datatype when possible
        np_dt = np.dtype(dtype)
        mpi_type = MPI._typedict[np_dt.char]

        if node_rank == 0:
            if self.is_master_rank:
                # Root node already has A
                shared_A[:] = root_array[:]

                # Send to other node leaders
                for r in range(1, leader_comm.Get_size()):
                    leader_comm.Send([root_array, mpi_type], dest=r)

            else:
                # Receive from global root leader
                leader_comm.Recv([shared_A, mpi_type], source=0)

        # Synchronize local ranks
        win.Fence()
        node_comm.Barrier()

        if get_win:
            return shared_A, win
        else:
            return shared_A

    def print_node_groups(self):
        r"""Print the world ranks grouped by node and identify each node leader.

        The output is emitted once per node by the node leader.
        """
        shared_comm = self.comm.Split_type(MPI.COMM_TYPE_SHARED, key=self.mpi_rank)
        shared_rank = shared_comm.Get_rank()
        shared_size = shared_comm.Get_size()

        sendbuf = np.array([self.mpi_rank], dtype=np.int32)
        recvbuf = np.empty(shared_size, dtype=np.int32)
        shared_comm.Allgather([sendbuf, MPI.INT], [recvbuf, MPI.INT])

        if shared_rank == 0:
            node_ranks = recvbuf.tolist()
            print(
                f"Node leader {self.mpi_rank}: ranks in node = {node_ranks}", flush=True
            )

        return recvbuf if shared_rank == 0 else None
