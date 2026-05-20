import numpy as np


class COMM_WORLD:
    def __init__(self):
        pass

    def bcast(self, data, **kwargs):
        return data

    def gather(self, data, **kwargs):
        return [data]


class MPI:
    SUM = None

    def __init__(self):
        self.COMM_WORLD = COMM_WORLD()


class SerialRoutine:
    def __init__(self, **kwargs):
        r"""This class provides the necessary placeholder attributes and methods
        for the serial version of the linear algebra routines.

        The additional arguments and keyword arguments of the functions are ignored,
        returning the default serial behaviour since they are only relevant for the
        parallel version of the distribution routine.
        """
        self.MPI = MPI()
        self.comm = self.MPI.COMM_WORLD
        self.mpi_rank = 0
        self.mpi_size = 1
        self.is_master_rank = True
        self.nprow, self.npcol = 1, 1

    def numroc(self, N, *args, **kwargs):
        return N

    def compute_indices(self, size, *args, **kwargs):
        return list(range(size)), list(range(size))

    def distribute(self, glob_A, *args, **kwargs):
        r"""Not needed for the serial version, since the matrix is not distributed."""
        return glob_A

    def distribute_diag(self, diag_values, *args, **kwargs):
        r"""Not needed for the serial version, since there is only one process.

        Parameters
        ----------
            diag_values : np.ndarray
                A 1D array containing the values to be placed on the diagonal of the
                global matrix.
        Returns
        -------
            np.ndarray
                A 2D array representing the diagonal matrix constructed from the input
                values, or the input array itself if it is already 2D.
        """
        if diag_values.ndim == 2:
            return diag_values
        else:
            return np.diag(diag_values)

    def get_diag(self, glob_A, *args, **kwargs):
        r"""Not needed for the serial version, since the matrix is not distributed."""
        if glob_A.ndim == 1:
            return glob_A
        else:
            return np.diag(glob_A)

    def gather(self, loc_A, *args, **kwargs):
        r"""Not needed for the serial version, since the matrix is not distributed."""
        return loc_A

    def shared_array(self, root_array: np.ndarray, *args, **kwargs):
        r"""Not needed for the serial version, since there is only one process."""
        return root_array

    def print_node_groups(self):
        r"""Print the world ranks grouped by node and identify each node leader.

        The output is emitted once per node by the node leader.
        """
        print("Only one process, so no node groups to print.")
