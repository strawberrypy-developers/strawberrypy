import numpy as np

from ..classes import Model
from .pair_distance import get_pbc_distance_pairs


def lattice_contraction(model: Model, cutoff: float) -> list:
    r"""Defines which atomic sites must be contracted on one site, for each
    site of the lattice.

    Parameters
    ----------
        model : Model
            The model for which to compute the lattice contraction.
        cutoff : float
            Real-space cutoff for the contraction window.

    Returns
    -------
        contraction : list
            List of the indices of the atoms that are within cutoff from each other,
            per atom.
    """
    contraction = []

    # Split the rows across ranks and build each local contraction block independently
    Nsites = model.cart_positions.shape[0]
    start = model.backend.mpi_rank * (Nsites // model.backend.mpi_size) + min(
        model.backend.mpi_rank, Nsites % model.backend.mpi_size
    )
    end = (
        start
        + (Nsites // model.backend.mpi_size)
        + (1 if model.backend.mpi_rank < (Nsites % model.backend.mpi_size) else 0)
    )

    local_contraction = []
    for current in range(start, end):
        dist = np.linalg.norm(
            model.cart_positions[current] - model.cart_positions, axis=1
        )

        local_contraction.append(np.flatnonzero(dist - cutoff <= 1e-6))

    gathered = model.backend.comm.gather(local_contraction, root=0)

    if model.backend.mpi_rank == 0:
        for part in gathered:
            contraction.extend(part)
    else:
        contraction = None

    contraction = model.backend.comm.bcast(contraction, root=0)

    return contraction


def pbc_lattice_contraction(
    model: Model, cutoff: float, distance_pairs: np.ndarray = None
) -> list:
    r"""Defines which atomic sites must be contracted on one site, for each
    site of the lattice within periodic boundary conditions (minimum image convention).

    Parameters
    ----------
        model : Model
            The model for which to compute the lattice contraction.
        cutoff : float
            Real-space cutoff for the contraction window.
        distance_pairs : np.ndarray, optional
            Precomputed pair distances for the model. If not provided, it will be computed
            using the :func:`get_pbc_distance_pairs` function. This can be useful if the
            pair distances have already been computed for other purposes, to avoid redundant
            calculations.

    Returns
    -------
        contraction : list
            List of the indices of the atoms that are within cutoff from each other,
            per atom.
    """
    if distance_pairs is None:
        distance_pairs = get_pbc_distance_pairs(model)

    # Parallelize by splitting the rows among MPI ranks and gathering
    N = distance_pairs.shape[0]
    start = model.backend.mpi_rank * (N // model.backend.mpi_size) + min(
        model.backend.mpi_rank, N % model.backend.mpi_size
    )
    end = (
        start
        + (N // model.backend.mpi_size)
        + (1 if model.backend.mpi_rank < (N % model.backend.mpi_size) else 0)
    )

    local_mask = distance_pairs[start:end] - cutoff <= 1e-8
    local_contraction = [np.flatnonzero(row) for row in local_mask]

    # Gather lists from all ranks to root
    gathered = model.backend.comm.gather(local_contraction, root=0)

    if model.backend.mpi_rank == 0:
        # Flatten the list of lists-of-arrays into a single list in correct order
        contraction = []
        for part in gathered:
            contraction.extend(part)
    else:
        contraction = None

    # Broadcast the full contraction to all ranks so everyone has the same data
    contraction = model.backend.comm.bcast(contraction, root=0)
    return contraction
