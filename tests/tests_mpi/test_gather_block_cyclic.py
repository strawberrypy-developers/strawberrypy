import numpy as np
import pytest
from strawberrypy.backends import get_backend


@pytest.mark.mpi
def test_gather_block_cyclic_roundtrip():
    nblk = 7
    backend, _ = get_backend(nblk=nblk)
    comm = backend.comm
    rank = backend.mpi_rank
    size = backend.mpi_size

    M, N = 9, 7

    if rank == 0:
        glob_A = np.arange(M * N, dtype=np.float64).reshape((M, N))
    else:
        glob_A = None

    loc_A = backend.distribute(glob_A, root=0, dtype=np.float64)

    # Reconstruct using the memory-efficient gather helper
    glob_rec = backend.gather(loc_A, M, N, root=0, dtype=np.float64)

    if rank == 0:
        assert np.array_equal(glob_rec, glob_A)
