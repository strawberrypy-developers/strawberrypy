from ..config import USE_MPI

# Detect if MPI is enabled and set up the appropriate backend for parallelization
if USE_MPI:
    from .mpi_linalg import MPILinalg
    from .mpi_physics import MPIPhysics

    def get_backend(**kwargs):
        return (MPILinalg(**kwargs), MPIPhysics(**kwargs))

else:
    from .serial_linalg import SerialLinalg
    from .serial_physics import SerialPhysics

    def get_backend(**kwargs):
        return (SerialLinalg(**kwargs), SerialPhysics(**kwargs))
