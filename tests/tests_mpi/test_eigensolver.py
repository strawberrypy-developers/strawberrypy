import numpy as np
import pytest
from strawberrypy.backends import get_backend
from strawberrypy.config import DEBUG_MODE


@pytest.mark.mpi
def test_eigensolver():
    nblk = 11
    backend, _ = get_backend(nblk=nblk)
    comm = backend.comm
    rank = backend.mpi_rank
    nprocs = backend.mpi_size

    dtype = np.complex128

    N = 1001

    if rank == 0:
        A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        # make it Hermitian
        H = (A + A.conj().T) / 2

    else:
        H = None

    loc_H = backend.distribute(glob_A=H, root=0, dtype=dtype)

    evals, evecs = backend.eigh(loc_H, N, N, collect_evec=True)

    if rank == 0:
        expected_evals, expected_evecs = np.linalg.eigh(H)
        assert np.allclose(evals, expected_evals)

        # Calculate the norm of the residual
        residual_norm = np.linalg.norm(H @ evecs - evecs @ np.diag(evals))
        if DEBUG_MODE:
            print(
                f"Norm of residual ||A*Z - Z*Lambda|| = {residual_norm:.6e}", flush=True
            )

        assert (
            residual_norm < 1e-9
        ), "Residual norm is too large, eigensolver may be incorrect."
