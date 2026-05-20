import numpy as np
import pytest
from strawberrypy.backends import get_backend


@pytest.mark.mpi
def test_distribute_diag_reconstructs_diagonal_matrix():
    nblk = 4
    backend, _ = get_backend(nblk=nblk)
    comm = backend.comm
    rank = backend.mpi_rank
    nprocs = backend.mpi_size
    dtype = np.complex128

    N = 12

    if rank == 0:
        diag_values = np.linspace(1.0, float(N), N, dtype=dtype)
    else:
        diag_values = None

    loc_A = backend.distribute_diag(diag_values=diag_values, root=0, dtype=dtype)

    gathered = comm.gather(loc_A, root=0)

    if rank == 0:
        reconstructed = np.zeros((N, N), dtype=dtype)

        for proc in range(nprocs):
            proc_row = proc // backend.npcol
            proc_col = proc % backend.npcol

            lri, gri = backend.compute_indices(N, backend.nprow, proc_row, nblk)
            lci, gci = backend.compute_indices(N, backend.npcol, proc_col, nblk)

            reconstructed[np.ix_(gri, gci)] = gathered[proc][np.ix_(lri, lci)]

        expected = np.diag(np.linspace(1.0, float(N), N, dtype=dtype))
        assert np.array_equal(reconstructed, expected)
