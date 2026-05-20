import numpy as np
import glob
import os

from ..classes import Model
from ..finite_model import FiniteModel
from ..supercell import Supercell

from .pair_distance import get_pbc_distance_pairs, get_obc_distance_pairs
from .contractions import pbc_lattice_contraction, lattice_contraction
from .averages import average_over_radius
from .marker_io import load_marker


def _all_distances(sup_model: Model) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute unique distances between a unit cell and all other cells in the supercell.

    The function returns a tuple containing a 1D array of unique distances and
    a 2D array of distances between all pairs of unit cells in the supercell.

    Parameters
    ----------
        sup_model : Model
            The supercell model for which to compute the distances.

    Returns
    -------
        unique_dist : np.ndarray
            A 1D array of unique distances between a unit cell and all the other cells in
            the supercell.
        distances : np.ndarray
            A 2D array of distances between all pairs of unit cells in the supercell.
    """
    if isinstance(sup_model, Supercell):
        distance_pairs = get_pbc_distance_pairs(sup_model)
    elif isinstance(sup_model, FiniteModel):
        distance_pairs = get_obc_distance_pairs(sup_model)

    # Indices are in terms of the orbitals of only one type. This is equivalent to taking
    #   the distances between the unit cell and all the other cells in the supercell
    distances = distance_pairs[:: sup_model.states_uc, :: sup_model.states_uc]
    unique_dist = np.sort(np.unique(np.round(distances[0, :], 7)))

    return unique_dist, np.round(distances, 7)


def _ivic_cell(
    sup_model: Model, unique_dist: np.ndarray, distances: np.ndarray
) -> np.ndarray:
    r"""Returns a 2D array where each row contains the indices of the two
    unit cells and the index of their distance in the `unique_dist` array.
    Expressed in terms of the orbitals of all types.

    Parameters
    ----------
        sup_model : Model
            The supercell model for which to compute the indices.
        unique_dist : np.ndarray
            1D array of unique distances between a unit cell and all other cells.
        distances : np.ndarray
            2D array of distances between all pairs of unit cells.

    Returns
    -------
        np.ndarray
            A 2D array of shape ``(N, 3)`` where N is the number of unit cell pairs.
            Each row contains ``(unit_cell_index_1, unit_cell_index_2, distance_index)``.
    """
    # Replace each element in the 2D array (distances) with the index in the 1D array
    #   (unique_dist) whose index value corresponds to the element value
    dist_indices = np.searchsorted(unique_dist, distances)

    # Express indices in terms of orbitals of only one type. This is equivalent to taking
    #   the indices of the unit cell pairs in the supercell
    row_idx, col_idx = np.indices(dist_indices.shape) * sup_model.states_uc
    ivic = np.column_stack((row_idx.ravel(), col_idx.ravel(), dist_indices.ravel()))

    return ivic


def correlation(
    sup_model: Model,
    directory: str | None = None,
    filenames: list[str] | str | None = None,
    is_bare_marker: bool = False,
    cutoff: float = 0.5,
    spinful: bool = False,
    to_normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Calculate the spatial correlation function of local markers over disorder realizations.

    This function computes the mean correlation over multiple disorder realizations
    (seeds). It averages the markers over a specified cutoff radius and calculates
    the normalized or unnormalized correlation function versus distance.

    Parameters
    ----------
        sup_model : Model
            The supercell model on which the markers were computed.
        directory : str
            Directory containing the marker data files.
        filenames : list of str or str
            Base filenames (prefix, without extension) of the data files.
        is_bare_marker : bool, optional
            Whether the marker data files contain the bare marker values (:python:`True`) or
            if they contain the marker values already averaged over the cutoff radius
            (:python:`False`). Default is :python:`False`.
        cutoff : float, optional
            Radius to consider for the local averaging (contraction) around each orbital if
            :python:`is_bare_marker == True`. Default is :python:`0.5`.
        spinful : bool, optional
            Whether the model and markers are spinful. If :python:`True`, computes the marker
            properly from spin-down and spin-up data columns. Default is :python:`False`.
        to_normalize : bool, optional
            Whether to normalize the correlation function by the variance.
            Default is :python:`True`.

    Returns
    -------
        unique_dist : np.ndarray
            1D array of unique distances.
        corr_r : np.ndarray
            1D array containing the mean correlation value for each corresponding distance.
    """
    if directory is None and filenames is None:
        raise ValueError("Either directory or filename must be provided.")
    elif directory is None and filenames is not None:
        # Process all the given files assuming their path is provided here
        file_list = filenames if isinstance(filenames, list) else [filenames]
    elif directory is not None and filenames is not None:
        # If filename is a string, we assume it's a common pattern to match files in the
        #   directory, else the list of files is provided directly
        if isinstance(filenames, str):
            file_list = glob.glob(f"{directory}/{filenames}*.npz")
        else:
            file_list = [f"{directory}/{fname}" for fname in filenames]
    elif directory is not None and filenames is None:
        # Process all files in the directory
        file_list = [f for f in os.listdir(directory) if f.endswith(".npz")]

    # Assert all the files have the right extension to be loaded
    file_list = [f if f.endswith(".npz") else f + ".npz" for f in file_list]
    if not file_list:
        raise ValueError("No matching files found to process.")

    Lx = sup_model.Lx
    Ly = sup_model.Ly

    # Same type of orbitals (i.e. unit cells in the supercell)
    unique_dist, distances = _all_distances(sup_model)
    ivic = _ivic_cell(sup_model, unique_dist, distances)

    # Contraction is a list of lists, each sublist contains the indices of the orbitals
    #   that are within the cutoff radius from the orbital of interest (which is the
    #   index of the sublist in the list)
    # Ex. contraction = [[0, 1, 2], [1, 0, 3], ...] means that the orbitals 0, 1, and 2
    #   are the closest to orbital 0; orbitals 1, 0, and 3 are the closest to orbital 1,
    #   etc. within the cutoff radius

    # All type of orbitals
    if isinstance(sup_model, Supercell):
        contraction = pbc_lattice_contraction(sup_model, cutoff=cutoff)
    elif isinstance(sup_model, FiniteModel):
        contraction = lattice_contraction(sup_model, cutoff=cutoff)

    corr_r = np.zeros(len(unique_dist))

    n_dist = np.array([np.sum(np.isclose(distances[0, :], val)) for val in unique_dist])

    if not spinful:
        for ctr, file in enumerate(file_list, start=1):
            print(f"Processing file: {file}", flush=True)
            c_i = load_marker("", file)[1]

            if is_bare_marker:
                marker_use = average_over_radius(
                    sup_model, c_i, cutoff=cutoff, contraction=contraction
                )
            else:
                marker_use = c_i

            # All type of orbitals
            marker0 = np.empty_like(marker_use)

            # Take the mean of the marker over the orbitals within unit cell in the
            #   supercell and replace the values with the mean.
            for t in range(0, len(marker_use), sup_model.states_uc):
                mean_t = np.mean(marker_use[t : t + sup_model.states_uc])
                marker0[t : t + sup_model.states_uc] = mean_t

            c_mean = np.mean(marker0[0 :: sup_model.states_uc])
            c2_mean = np.mean(marker0[0 :: sup_model.states_uc] ** 2)

            if to_normalize:
                var_c = c2_mean - c_mean**2
            else:
                var_c = 1.0

            corr_r += np.bincount(
                ivic[:, 2],
                weights=(marker0[ivic[:, 0]] * marker0[ivic[:, 1]] - c_mean**2) / var_c,
                minlength=len(corr_r),
            )

    elif spinful:
        for ctr, file in enumerate(file_list, start=1):
            print(f"Processing file: {file}", flush=True)
            c_i_p = load_marker("", file)[1][0]
            c_i_m = load_marker("", file)[1][1]

            if is_bare_marker:
                marker_use_p = average_over_radius(
                    sup_model, c_i_p, cutoff=cutoff, contraction=contraction
                )

                marker_use_m = average_over_radius(
                    sup_model, c_i_m, cutoff=cutoff, contraction=contraction
                )

                marker_use = np.mod(0.5 * (marker_use_p - marker_use_m), 2)
                marker_use = 1 - np.abs(1 - marker_use)

            else:
                marker_use = np.mod(0.5 * (c_i_p - c_i_m), 2)
                marker_use = 1 - np.abs(1 - marker_use)

            # All type of orbitals
            marker0 = np.empty_like(marker_use)

            # Take the mean of the marker over the orbitals within unit cell in the
            #   supercell and replace the values with the mean.
            for t in range(0, len(marker_use), sup_model.states_uc):
                mean_t = np.mean(marker_use[t : t + sup_model.states_uc])
                marker0[t : t + sup_model.states_uc] = mean_t

            c_mean = np.mean(marker0[0 :: sup_model.states_uc])
            c2_mean = np.mean(marker0[0 :: sup_model.states_uc] ** 2)

            if to_normalize:
                var_c = c2_mean - c_mean**2
            else:
                var_c = 1.0

            corr_r += np.bincount(
                ivic[:, 2],
                weights=(marker0[ivic[:, 0]] * marker0[ivic[:, 1]] - c_mean**2) / var_c,
                minlength=len(corr_r),
            )

    corr_r /= n_dist
    corr_r /= ctr
    corr_r /= Lx * Ly

    return unique_dist, corr_r
