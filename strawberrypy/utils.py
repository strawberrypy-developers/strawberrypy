r"""A module containing utility functions for internal use that are not related to Models specifically."""

import numpy as np


def unique_vacancies(
    num: int,
    Lx: int,
    Ly: int,
    basis: int,
    atom_type: int = None,
    same_number: bool = False,
    seed: int = None,
):
    r"""Returns a list of unique random lattice sites to be removed in the model
    using the method :meth:`~strawberrypy.Model.add_vacancies`.

    Parameters
    ----------
        num :
            Number of lattice positions to generate.
        Lx :
            Number of unit cells of the model along the :math:`\mathbf{a}_1` direction.
        Ly :
            Number of unit cells of the model along the :math:`\mathbf{a}_2` direction.
        basis :
            Number of atoms per unit cell.
        atom_type :
            Index of the atom to be removed.
        same_number :
            If :python:`True`, the same number of each type of atom is removed.
        seed :
            Seed for the random number generation. Default is :python:`None`.

    Returns
    -------
        unique_list : :python:`list`
            List of unique random lattice site.
    """
    indexes = []
    unique_list = []

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    if same_number:
        atom = np.zeros(num, dtype=np.integer)
        type_B = np.zeros(num, dtype=np.integer)
        type_B[: num // basis] = np.ones(num // basis, dtype=np.integer)

        nonzeros = np.sort(np.nonzero(np.random.permutation(type_B))[0])
        atom[nonzeros] = np.ones(num // basis, dtype=np.integer)

        ind = 0
        # Generate num unique entries
        while len(unique_list) < num:
            # Trial entry
            at_type = atom[ind]

            trial = [np.random.randint(Lx), np.random.randint(Ly), at_type]

            # Generate internal index
            trial_index = Ly * basis * trial[0] + basis * trial[1] + trial[2]

            # If this is a new entry store it
            if trial_index not in indexes:
                indexes.append(trial_index)
                unique_list.append(trial)

                ind += 1
    else:
        # Generate num unique entries
        while len(unique_list) < num:
            # Trial entry
            if atom_type is not None:
                trial = [np.random.randint(Lx), np.random.randint(Ly), atom_type]
            else:
                trial = [
                    np.random.randint(Lx),
                    np.random.randint(Ly),
                    np.random.randint(basis),
                ]

            # Generate internal index
            trial_index = Ly * basis * trial[0] + basis * trial[1] + trial[2]

            # If this is a new entry store it
            if trial_index not in indexes:
                indexes.append(trial_index)
                unique_list.append(trial)

    return unique_list


def _create_orb_pattern(orb_pos: list[list[float]]) -> list[int]:
    r"""Creates a pattern list that assigns the same index (starting from 0) to orbitals
    that occupy the same position within the unit cell, within a specified tolerance.

    Parameters
    ----------
    orb_pos : list of list of float
        A list of orbital positions, where each position is a list of three floats
        representing the (x, y, z) coordinates of the orbital within the unit cell.

    Returns
    -------
    pattern : list of int
        A list of integers where each integer corresponds to the index of the unique
        position that the orbital occupies. Orbitals that occupy the same position
        (within the specified tolerance) will have the same index.

    Examples
    -------
    >>> orb_pos = [[0.0, 0.0], [0.00, 0.], [0.123, 0.24], [0.23, 0.25], [0.230, 0.250]]
    >>> _create_orb_pattern(orb_pos)
    [0, 0, 1, 2, 2]
    """

    tolerance = 1e-7
    pattern = []
    unique_points = []
    counter = 0

    for pos in orb_pos:
        found = False
        for i, upos in enumerate(unique_points):
            # Compare both coordinates with tolerance
            if (
                abs(pos[0] - upos[0]) < tolerance
                and abs(pos[1] - upos[1]) < tolerance
                and abs(pos[2] - upos[2]) < tolerance
            ):
                pattern.append(i)
                found = True
                break
        if not found:
            unique_points.append(pos)
            pattern.append(counter)
            counter += 1

    return pattern
