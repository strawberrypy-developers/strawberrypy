import pytest


#  Mark this test so it ONLY runs if we use the --with-mpi flag
@pytest.mark.mpi
def test_size():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    assert comm.size > 0


@pytest.mark.mpi
def test_broadcast_data():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        data = {"key": "value"}
    else:
        data = None

    # Broadcast data from Rank 0 to all others
    data = comm.bcast(data, root=0)

    # Assert works on all ranks
    assert data["key"] == "value"


#  You can also skip specific tests on specific ranks (rare but useful)
@pytest.mark.mpi
def test_only_rank_zero():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank != 0:
        return  # Simply return to pass on other ranks

    # Perform specific Rank 0 checks
    assert rank == 0
