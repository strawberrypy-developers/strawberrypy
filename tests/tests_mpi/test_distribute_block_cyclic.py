import numpy as np
import pytest
from strawberrypy.backends import get_backend


@pytest.mark.mpi
def test_distribute_block_cyclic_reconstructs_global_matrix():
    nblk = 4
    backend, _ = get_backend(nblk=nblk)

    comm = backend.comm
    rank = backend.mpi_rank
    nprocs = backend.mpi_size

    M, N = 7, 5

    if rank == 0:
        glob_A = np.arange(M * N, dtype=np.float64).reshape((M, N))
    else:
        glob_A = None

    loc_A = backend.distribute(glob_A=glob_A, root=0, dtype=np.float64)

    nprow = int(np.floor(np.sqrt(nprocs)))
    while nprocs % nprow != 0:
        nprow -= 1
    npcol = nprocs // nprow

    myprow = rank // npcol
    mypcol = rank % npcol

    exp_lr = backend.numroc(M, nblk, myprow, 0, nprow)
    exp_lc = backend.numroc(N, nblk, mypcol, 0, npcol)
    assert loc_A.shape == (exp_lr, exp_lc)

    gathered = comm.gather(loc_A, root=0)

    if rank == 0:
        reconstructed = np.empty((M, N), dtype=np.float64)

        for proc in range(nprocs):
            proc_row = proc // npcol
            proc_col = proc % npcol

            lri, gri = backend.compute_indices(M, nprow, proc_row, nblk)
            lci, gci = backend.compute_indices(N, npcol, proc_col, nblk)
            reconstructed[np.ix_(gri, gci)] = gathered[proc][np.ix_(lri, lci)]

        assert np.array_equal(reconstructed, glob_A)
