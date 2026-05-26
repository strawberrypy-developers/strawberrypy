import numpy as np
import pytest
from strawberrypy.backends import get_backend
from strawberrypy.config import DEBUG_MODE


@pytest.mark.mpi
def test_get_diag_broadcast_roundtrip():
    nblk = 3
    backend, _ = get_backend(nblk=nblk)
    comm = backend.comm
    rank = backend.mpi_rank

    N = 11

    if rank == 0:
        glob_A = np.arange(N * N, dtype=np.float64).reshape((N, N))
    else:
        glob_A = None

    loc_A = backend.distribute(glob_A, root=0, dtype=np.float64)

    # Use broadcast=True to receive the diagonal on all ranks
    diag_all = backend.get_diag(loc_A, N, root=0, dtype=np.float64, broadcast=True)

    if DEBUG_MODE and rank == 0:
        print(f"Rank {rank}: diag_all={diag_all}", flush=True)

    # On every rank diag_all should be available
    assert diag_all is not None

    # Build expected diagonal on root, then broadcast once to all ranks
    expected = np.diag(glob_A) if rank == 0 else None
    expected = comm.bcast(expected, root=0)

    # Check shape
    assert diag_all.shape == (N,)

    # Verify values on all ranks
    assert np.array_equal(diag_all, expected)
