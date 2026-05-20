import numpy as np
import pytest
from strawberrypy.backends import get_backend


@pytest.mark.mpi
def test_matmul():
    nblk = 10
    backend, _ = get_backend(nblk=nblk)
    comm = backend.comm
    rank = backend.mpi_rank
    nprocs = backend.mpi_size

    dtype = np.complex128

    M, K, N = 101, 202, 303

    if rank == 0:
        glob_A = np.arange(M * K, dtype=dtype).reshape((M, K))
        glob_B = np.arange(K * N, dtype=dtype).reshape((K, N)) * 1j
    else:
        glob_A = None
        glob_B = None

    loc_A = backend.distribute(glob_A=glob_A, root=0, dtype=dtype)
    loc_B = backend.distribute(glob_A=glob_B, root=0, dtype=dtype)

    loc_C = backend.matmul(loc_A, loc_B, (M, K), (K, N))

    glob_C = backend.gather(loc_C, M, N, dtype=dtype)

    if rank == 0:
        expected_C = glob_A @ glob_B
        assert np.allclose(glob_C, expected_C)
