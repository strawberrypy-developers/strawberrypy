r"""Configuration and global variables for strawberrypy"""

from functools import wraps
import os

####################################################################################
# Global version of the package
__version__ = "0.4.0"

# Enable or disable debug mode (for testing purposes)
DEBUG_MODE = bool(int(os.environ.get("STRAWBERRYPY_DEBUG_MODE", "0")))
####################################################################################

# Detect MPI parallelization
try:
    from mpi4py import MPI

    mpi_size = MPI.COMM_WORLD.Get_size()
    if not mpi_size > 1 and not DEBUG_MODE:
        raise ImportError("Effectively serial execution")

    USE_MPI = True

except (ImportError, RuntimeError):
    # ImportError: mpi4py is not installed or cannot be imported
    # RuntimeError: an installation of mpi4py is present but cannot be initialized
    #   properly (e.g., missing shared libraries, incompatible MPI, ...)

    USE_MPI = False


# Decorator to avoid returning values on ranks different from master
def mpimaster(func):
    r"""
    Force a return value only on the master rank in an MPI parallelized code.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        x = func(*args, **kwargs)

        try:
            is_master = args[0].backend.is_master_rank
        except Exception:
            is_master = False

        if is_master:
            return x
        else:
            # If the wrapped function returned multiple values (tuple/list),
            #   return the same-shaped sequence filled with None on non-master ranks
            if isinstance(x, tuple):
                return tuple(None for _ in x)
            if isinstance(x, list):
                return [None for _ in x]
            return None

    return wrapper
