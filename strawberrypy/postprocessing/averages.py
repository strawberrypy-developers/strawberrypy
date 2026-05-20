import numpy as np

from ..config import USE_MPI
from ..classes import Model
from .contractions import lattice_contraction


def average_over_radius(
    model: Model, vals: np.ndarray, cutoff: float, contraction: list = None
) -> np.ndarray:
    r"""Average of values in real space according to a certain cutoff.

    Parameters
    ----------
        model : Model
            The model for which to compute the average.
        vals : array-like
            Values that have to be averaged.
        cutoff : float
            Real space cutoff for the real space average.
        contraction : list, optional
            List of atoms that have to be considered in the macroscopic average, per
            atom. Default is :python:`None`, which means it is computed from the
            positions of the atoms in the lattice assuming open boundary conditions.

    Returns
    -------
        return_vals : np.ndarray
            List of averaged values per lattice site within a cutoff.
    """
    return_vals = []

    # Macroscopic average within a certain radius = cutoff
    # If no contraction list is passed, compute it (OBC default)
    if contraction is None:
        contraction = lattice_contraction(model, cutoff)

    # Number of sites
    Nsites = len(contraction)

    # Split work among MPI ranks: contiguous block per rank
    start = model.backend.mpi_rank * (Nsites // model.backend.mpi_size) + min(
        model.backend.mpi_rank, Nsites % model.backend.mpi_size
    )
    end = (
        start
        + (Nsites // model.backend.mpi_size)
        + (1 if model.backend.mpi_rank < (Nsites % model.backend.mpi_size) else 0)
    )

    # Local contributions (aligned to global indices)
    local_sums = np.zeros(Nsites, dtype=float)
    local_counts = np.zeros(Nsites, dtype=float)

    for i in range(start, end):
        neigh = contraction[i]
        local_sums[i] = np.sum(vals[neigh])
        local_counts[i] = len(neigh)

    # Reduce to global sums and counts
    if USE_MPI:
        global_sums = np.zeros_like(local_sums)
        global_counts = np.zeros_like(local_counts)
        model.backend.comm.Allreduce(local_sums, global_sums)
        model.backend.comm.Allreduce(local_counts, global_counts)
    else:
        global_sums = local_sums
        global_counts = local_counts

    # Check for zero neighbor counts
    if np.any(global_counts == 0):
        raise RuntimeError("Unexpected error: some lattice sites have no neighbors.")

    # Compute averaged values
    return_vals = np.where(
        global_counts <= model.states_uc,
        global_sums,  # When the macroscopic average is within a radius smaller than the unit cell
        # Else normalized average
        global_sums / (global_counts / model.states_uc),
    )

    return np.array(return_vals)
