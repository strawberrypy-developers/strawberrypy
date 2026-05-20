import numpy as np

from ..config import USE_MPI
from ..classes import Model


def shift_x(model: Model, orb_arr: np.ndarray) -> np.ndarray:
    r"""
    Shift the whole lattice to the right. Only used for the PBC lattice contraction.

    Parameters
    ----------
        model: Model
            The model containing the lattice parameters.
        orb_arr :
            Site indices which then are mapped out during transformation. The lattice
            is labeled where y-coords vary first.

    Returns
    -------
        shifted_orb_arr : np.ndarray
            The shifted site indices after transformation.
    """
    return (orb_arr + model.states_uc * model.Ly) % (
        model.Lx * model.Ly * model.states_uc
    )


def shift_y(model: Model, orb_arr: np.ndarray) -> np.ndarray:
    r"""
    Shift the whole lattice upwards.

    Parameters
    ----------
        model: Model
            The model containing the lattice parameters.
        orb_arr :
            Site indices which then are mapped out during transformation. The lattice
            is labeled where y-coords vary first.

    Returns
    -------
        shifted_orb_arr : np.ndarray
            The shifted site indices after transformation.
    """
    nstates_per_col = model.states_uc * model.Ly
    col_index = (orb_arr // nstates_per_col) * nstates_per_col
    row_index = (orb_arr % nstates_per_col + model.states_uc) % nstates_per_col
    return col_index + row_index


def _get_pbc_distance_pairs_amorphous_serial(model: Model) -> np.ndarray:
    r"""Compute the distance between pair of orbitals with periodic boundary
    conditions, for an amorphous system.

    Parameters
    ----------
        model : Model
            The model for which to compute the pair distances.

    Returns
    -------
        distance_pairs : np.ndarray
            The matrix of pair distances in cartesian coordinates, accounting for periodic
            boundary conditions.
    """
    lvecs = model.lvecs

    # Reduced Lvecs
    lvecs[0] /= model.Lx
    lvecs[1] /= model.Ly

    cart_positions = model.cart_positions

    # Compute reduced coordinates
    linv = np.linalg.inv(lvecs)
    reds = cart_positions @ linv
    N = reds.shape[0]

    distance_pairs = np.zeros((N, N))

    for i in range(N):
        for trial in range(i + 1, N):
            # Periodic images of trial orbital in reduced coordinates
            trial_orbs = reds[trial] + [
                (0, 0, 0),
                (model.Lx, 0, 0),
                (-model.Lx, 0, 0),
                (0, model.Ly, 0),
                (0, -model.Ly, 0),
                (model.Lx, model.Ly, 0),
                (-model.Lx, -model.Ly, 0),
                (-model.Lx, model.Ly, 0),
                (model.Lx, -model.Ly, 0),
            ]
            # Find the minimum (cartesian) distance among all periodic images to orbital i
            dist = np.linalg.norm((trial_orbs - reds[i]) @ lvecs, axis=1).min()
            distance_pairs[i, trial] = dist
            distance_pairs[trial, i] = dist

    # Round values as to avoid very near values being tagged as separate distances.
    distance_pairs = np.round(distance_pairs, 7)

    return distance_pairs


def _get_pbc_distance_pairs_amorphous_mpi(model: Model) -> np.ndarray:
    r"""Compute the distance between pair of orbitals with periodic boundary
    conditions, for an amorphous system.

    Parameters
    ----------
        model : Model
            The model for which to compute the pair distances.

    Returns
    -------
        distance_pairs : np.ndarray
            The matrix of pair distances in cartesian coordinates, accounting for periodic
            boundary conditions.
    """
    lvecs = model.backend.comm.bcast(model.lvecs, root=0)

    # Reduced Lvecs
    lvecs[0] /= model.Lx
    lvecs[1] /= model.Ly

    cart_positions = model.cart_positions

    # Compute reduced coordinates
    linv = np.linalg.inv(lvecs)
    reds = cart_positions @ linv
    N = reds.shape[0]

    block_size = max(1, N // model.backend.mpi_size)
    start_i = model.backend.mpi_rank * block_size
    if model.backend.mpi_rank == model.backend.mpi_size - 1:
        end_i = N
    else:
        end_i = (model.backend.mpi_rank + 1) * block_size

    start_i = min(max(start_i, 0), N)
    end_i = min(max(end_i, start_i), N)
    local_rows = end_i - start_i

    local_dist_uc = np.zeros((local_rows, N), dtype=np.float64)

    for i in range(start_i, end_i):
        for trial in range(i + 1, N):
            # Periodic images of trial orbital in reduced coordinates
            trial_orbs = reds[trial] + [
                (0, 0, 0),
                (model.Lx, 0, 0),
                (-model.Lx, 0, 0),
                (0, model.Ly, 0),
                (0, -model.Ly, 0),
                (model.Lx, model.Ly, 0),
                (-model.Lx, -model.Ly, 0),
                (-model.Lx, model.Ly, 0),
                (model.Lx, -model.Ly, 0),
            ]
            # Find the minimum (cartesian) distance among all periodic images to orbital i
            dist = np.linalg.norm((trial_orbs - reds[i]) @ lvecs, axis=1).min()
            local_dist_uc[i - start_i, trial] = dist

    if model.backend.is_master_rank:
        global_dist = np.zeros((N, N), dtype=np.float64)
        if local_rows > 0:
            global_dist[start_i:end_i, :] = local_dist_uc

        for r in range(1, model.backend.mpi_size):
            r_start = r * block_size
            r_end = N if r == model.backend.mpi_size - 1 else (r + 1) * block_size
            r_start = min(max(r_start, 0), N)
            r_end = min(max(r_end, r_start), N)

            if r_end > r_start:
                temp = np.empty((r_end - r_start, N), dtype=np.float64)
                model.backend.comm.Recv(temp, source=r, tag=0)
                global_dist[r_start:r_end, :] = temp

        # Symmetrize the upper triangular matrix
        global_dist = global_dist + global_dist.T
    else:
        global_dist = None
        if local_rows > 0:
            model.backend.comm.Send(local_dist_uc, dest=0, tag=0)

    del local_dist_uc  # Free memory as it's no longer needed

    distance_pairs, win = model.backend.shared_array(global_dist, get_win=True)

    # Use node comm so only 1 rank per physical node executes the duplicate array rounding
    node_comm = model.backend.comm.Split_type(model.backend.MPI.COMM_TYPE_SHARED)
    if node_comm.Get_rank() == 0:
        np.round(distance_pairs, 7, out=distance_pairs)
    win.Fence()

    return distance_pairs


def _get_pbc_distance_pairs_serial(model: Model) -> np.ndarray:
    r"""Compute the distance between pair of orbitals with periodic boundary
    conditions.

    Parameters
    ----------
        model : Model
            The model for which to compute the pair distances.

    Returns
    -------
        distance_pairs : np.ndarray
            The matrix of pair distances in cartesian coordinates, accounting for periodic
            boundary conditions (minimum convention image).
    """
    lvecs = model.lvecs

    # Reduced Lvecs
    lvecs[0] /= model.Lx
    lvecs[1] /= model.Ly

    if model.has_vacancies:
        cart_positions = model._cart_positions_original
    else:
        cart_positions = model.cart_positions

    # Compute reduced coordinates
    linv = np.linalg.inv(lvecs)
    reds = cart_positions @ linv
    N = reds.shape[0]

    distance_pairs = np.zeros((N, N))

    # Assume that cart_positions are arranged such that the y-coords vary first for a
    #   given x-coords
    # Compute first distances w.r.t to home unit cell. Take advantage that translating
    #   the whole lattice will not change the pair distance
    # Indices are in terms of all the orbitals
    for i in range(model.states_uc):
        for trial in range(i + 1, N):
            # Periodic images of trial orbital in reduced coordinates
            trial_orbs = reds[trial] + [
                (0, 0, 0),
                (model.Lx, 0, 0),
                (-model.Lx, 0, 0),
                (0, model.Ly, 0),
                (0, -model.Ly, 0),
                (model.Lx, model.Ly, 0),
                (-model.Lx, -model.Ly, 0),
                (-model.Lx, model.Ly, 0),
                (model.Lx, -model.Ly, 0),
            ]
            # Find the minimum (cartesian) distance among all periodic images to orbital i
            dist = np.linalg.norm((trial_orbs - reds[i]) @ lvecs, axis=1).min()
            distance_pairs[i, trial] = dist
            distance_pairs[trial, i] = dist

    orb_arr = np.arange(0, model.Lx * model.Ly * model.states_uc)
    shift = np.copy(orb_arr)

    # Lattice shift
    for _ in range(model.Ly):
        for _ in range(model.Lx):
            shift = np.copy(shift_x(model, shift))
            for i in range(model.states_uc):
                distance_pairs[shift[i], shift] = distance_pairs[orb_arr[i], orb_arr]

        shift = np.copy(shift_y(model, shift))
        for i in range(model.states_uc):
            distance_pairs[shift[i], shift] = distance_pairs[orb_arr[i], orb_arr]

    # Round values as to avoid very near values being tagged as separate distances.
    distance_pairs = np.round(distance_pairs, 7)
    # Remove distances involving vacancies, if any
    distance_pairs = distance_pairs[:, model._mask][model._mask, :]

    return distance_pairs


def _get_pbc_distance_pairs_mpi(model: Model) -> np.ndarray:
    r"""Compute the distance between pair of orbitals with periodic boundary
    conditions.

    Parameters
    ----------
        model : Model
            The model for which to compute the pair distances.

    Returns
    -------
        distance_pairs : np.ndarray
            The matrix of pair distances in cartesian coordinates, accounting for periodic
            boundary conditions (minimum convention image).
    """
    lvecs = model.backend.comm.bcast(model.lvecs, root=0)

    # Reduced Lvecs
    lvecs[0] /= model.Lx
    lvecs[1] /= model.Ly

    if model.has_vacancies:
        cart_positions = model._cart_positions_original
    else:
        cart_positions = model.cart_positions

    # Compute reduced coordinates
    linv = np.linalg.inv(lvecs)
    reds = cart_positions @ linv
    N = reds.shape[0]

    num_trials = N
    block_size = max(1, num_trials // model.backend.mpi_size)
    start_trial = model.backend.mpi_rank * block_size
    if model.backend.mpi_rank == model.backend.mpi_size - 1:
        end_trial = N
    else:
        end_trial = (model.backend.mpi_rank + 1) * block_size

    # Clamp start_trial, end_trial if they go out of bounds
    start_trial = min(max(start_trial, 0), N)
    end_trial = min(max(end_trial, start_trial), N)
    local_cols = end_trial - start_trial

    # Memory trick: compute only the local column slice of the upper band per rank
    local_dist_uc = np.zeros((model.states_uc, local_cols), dtype=np.float64)

    # Parallelize the initial pair distance computation
    for i in range(model.states_uc):
        for trial in range(start_trial, end_trial):
            # Periodic images of trial orbital in reduced coordinates
            trial_orbs = reds[trial] + [
                (0, 0, 0),
                (model.Lx, 0, 0),
                (-model.Lx, 0, 0),
                (0, model.Ly, 0),
                (0, -model.Ly, 0),
                (model.Lx, model.Ly, 0),
                (-model.Lx, -model.Ly, 0),
                (-model.Lx, model.Ly, 0),
                (model.Lx, -model.Ly, 0),
            ]
            # Find the minimum (cartesian) distance among all periodic images to orbital i
            dist = np.linalg.norm((trial_orbs - reds[i]) @ lvecs, axis=1).min()
            local_dist_uc[i, trial - start_trial] = dist

    # Gather explicitly onto the master rank
    if model.backend.is_master_rank:
        global_dist_uc = np.zeros((model.states_uc, N), dtype=np.float64)
        if local_cols > 0:
            global_dist_uc[:, start_trial:end_trial] = local_dist_uc

        for r in range(1, model.backend.mpi_size):
            r_start = r * block_size
            r_end = N if r == model.backend.mpi_size - 1 else (r + 1) * block_size
            r_start = min(max(r_start, 0), N)
            r_end = min(max(r_end, r_start), N)

            if r_end > r_start:
                temp = np.empty((model.states_uc, r_end - r_start), dtype=np.float64)
                model.backend.comm.Recv(temp, source=r, tag=0)
                global_dist_uc[:, r_start:r_end] = temp

        distance_pairs = np.zeros((N, N), dtype=np.float64)
        for i in range(model.states_uc):
            distance_pairs[i, :] = global_dist_uc[i, :]
            distance_pairs[:, i] = global_dist_uc[i, :]
    else:
        distance_pairs = None
        if local_cols > 0:
            model.backend.comm.Send(local_dist_uc, dest=0, tag=0)

    del local_dist_uc  # Free memory as it's no longer needed

    distance_pairs, win = model.backend.shared_array(distance_pairs, get_win=True)

    #  Only node rank leaders write shared array
    node_comm = model.backend.comm.Split_type(model.backend.MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()

    # Lattice shift - distributed across node internal ranks for speed without memory footprint
    orb_arr = np.arange(0, N)
    shift = np.copy(orb_arr)

    for _ in range(model.Ly):
        for _ in range(model.Lx):
            shift = np.copy(shift_x(model, shift))
            for i in range(model.states_uc):
                if i % node_comm.size == node_rank:
                    distance_pairs[shift[i], shift] = distance_pairs[orb_arr[i], orb_arr]
            win.Fence()

        shift = np.copy(shift_y(model, shift))
        for i in range(model.states_uc):
            if i % node_comm.size == node_rank:
                distance_pairs[shift[i], shift] = distance_pairs[orb_arr[i], orb_arr]
        win.Fence()

    if node_rank == 0:
        # Round values as to avoid very near values being tagged as separate distances
        np.round(distance_pairs, 7, out=distance_pairs)
    win.Fence()

    # Remove distances involving vacancies, if any
    distance_pairs = distance_pairs[:, model._mask][model._mask, :]

    return distance_pairs


def _get_obc_distance_pairs_serial(model: Model) -> np.ndarray:
    r"""Compute the distance between pair of orbitals with open boundary
    conditions.

    Parameters
    ----------
        model : Model
            The model for which to compute the pair distances.

    Returns
    -------
        distance_pairs : np.ndarray
            The matrix of pair distances in cartesian coordinates, without accounting for
            periodic boundary conditions.
    """

    if model.has_vacancies:
        cart_positions = model._cart_positions_original
    else:
        cart_positions = model.cart_positions

    N = cart_positions.shape[0]

    distance_pairs = np.zeros((N, N))

    for i in range(N):
        dist = np.linalg.norm(cart_positions[i] - cart_positions[i + 1 : N], axis=1)
        distance_pairs[i + 1 : N, i] = dist

    # Symmetrize the upper triangular matrix
    distance_pairs = distance_pairs + distance_pairs.T

    # Round values as to avoid very near values being tagged as separate distances.
    distance_pairs = np.round(distance_pairs, 7)

    # Remove distances involving vacancies, if any
    distance_pairs = distance_pairs[:, model._mask][model._mask, :]

    return distance_pairs


def _get_obc_distance_pairs_mpi(model: Model) -> np.ndarray:
    r"""Compute the distance between pair of orbitals with open boundary
    conditions.

    Parameters
    ----------
        model : Model
            The model for which to compute the pair distances.

    Returns
    -------
        distance_pairs : np.ndarray
            The matrix of pair distances in cartesian coordinates, without accounting for
            periodic boundary conditions.
    """

    if model.has_vacancies:
        cart_positions = model._cart_positions_original
    else:
        cart_positions = model.cart_positions

    N = cart_positions.shape[0]

    block_size = max(1, N // model.backend.mpi_size)
    start_i = model.backend.mpi_rank * block_size
    if model.backend.mpi_rank == model.backend.mpi_size - 1:
        end_i = N
    else:
        end_i = (model.backend.mpi_rank + 1) * block_size

    start_i = min(max(start_i, 0), N)
    end_i = min(max(end_i, start_i), N)
    local_rows = end_i - start_i

    local_dist = np.zeros((local_rows, N), dtype=np.float64)

    for i in range(start_i, end_i):
        dist = np.linalg.norm(cart_positions[i] - cart_positions[i + 1 : N], axis=1)
        local_dist[i - start_i, i + 1 : N] = dist

    if model.backend.is_master_rank:
        global_dist = np.zeros((N, N), dtype=np.float64)
        if local_rows > 0:
            global_dist[start_i:end_i, :] = local_dist

        for r in range(1, model.backend.mpi_size):
            r_start = r * block_size
            r_end = N if r == model.backend.mpi_size - 1 else (r + 1) * block_size
            r_start = min(max(r_start, 0), N)
            r_end = min(max(r_end, r_start), N)

            if r_end > r_start:
                temp = np.empty((r_end - r_start, N), dtype=np.float64)
                model.backend.comm.Recv(temp, source=r, tag=0)
                global_dist[r_start:r_end, :] = temp

        # Symmetrize the upper triangular matrix
        global_dist = global_dist + global_dist.T
    else:
        global_dist = None
        if local_rows > 0:
            model.backend.comm.Send(local_dist, dest=0, tag=0)

    del local_dist

    distance_pairs, win = model.backend.shared_array(global_dist, get_win=True)

    # Use node comm so only 1 rank per physical node executes the duplicate array rounding
    node_comm = model.backend.comm.Split_type(model.backend.MPI.COMM_TYPE_SHARED)
    if node_comm.Get_rank() == 0:
        np.round(distance_pairs, 7, out=distance_pairs)
    win.Fence()

    # Remove distances involving vacancies, if any
    distance_pairs = distance_pairs[:, model._mask][model._mask, :]

    return distance_pairs


##########################################################################################

if USE_MPI:

    def get_pbc_distance_pairs(model: Model) -> np.ndarray:
        if model.is_crystalline:
            return _get_pbc_distance_pairs_mpi(model)
        else:
            return _get_pbc_distance_pairs_amorphous_mpi(model)

    def get_obc_distance_pairs(model: Model) -> np.ndarray:
        return _get_obc_distance_pairs_mpi(model)

else:

    def get_pbc_distance_pairs(model: Model) -> np.ndarray:
        if model.is_crystalline:
            return _get_pbc_distance_pairs_serial(model)
        else:
            return _get_pbc_distance_pairs_amorphous_serial(model)

    def get_obc_distance_pairs(model: Model) -> np.ndarray:
        return _get_obc_distance_pairs_serial(model)
