import numpy as np


def _orb_cart(model):
    r"""Returns the cartesian coordinates of the orbitals of a model.

    Parameters
    ----------
        model :
            A ``pythtb`` model instance.
        nx_sites :
            Number of unit cells in the model along the :math:`\mathbf{a}_1` direction.
        ny_sites :
            Number of unit cells in the model along the :math:`\mathbf{a}_2` direction.

    Returns
    -------
        positions :
            Cartesian coordinates of the lattice sites.

    .. warning::
        This function is meant for internal use only since it does not discriminate whether
        the model is spinful or not. The use of ``get_positions`` should be preferred.
    """
    try:
        # Raises AttributeError if the model is not pythtb.TBModel
        n_orb = model.norb
        lat_super = model.get_lat_vecs()
        orb_red = model.get_orb_vecs()
    except AttributeError:
        n_orb = model.get_num_orbitals()
        lat_super = model.get_lat()
        orb_red = model.get_orb()

    orb_c = []
    for i in range(n_orb):
        orb_c.append(
            (np.matmul(lat_super.transpose(), orb_red[i].reshape(-1, 1))).squeeze()
        )
    return np.array(orb_c)


def _reciprocal_vec(model):
    r"""Returns reciprocal lattice vectors in cartesian coordinates.

    Parameters
    ----------
        model :
            A ``pythtb`` model instance.

    Returns
    -------
        b1, b2 :
            Reciprocal lattice vectors.
    """
    try:
        try:
            bvecs = model.recip_lat_vecs
        except ValueError:
            bvecs = [None for _ in range(model.dim_r)]  # Case for finite systems
    except AttributeError:
        # Raises AttributeError if the model is not pythtb.tb_model
        _ = model._dim_k
        bvecs = (2.0 * np.pi) * np.linalg.inv(model.get_lat()).T

    return bvecs


def get_positions(model, spinful):
    r"""Returns the cartesian coordinates of the orbitals of a model.

    Parameters
    ----------
        model :
            A ``pythtb`` model instance.
        spinful :
            Whether the model is spinful or not.

    Returns
    -------
        positions :
            Cartesian coordinates of the lattice sites.
    """
    if not spinful:
        return _orb_cart(model)
    else:
        return np.repeat(_orb_cart(model), 2, axis=0)


def get_hamiltonian(model, spinful, point, dim):
    r"""Returns the Hamiltonian at the given k-point.

    Parameters
    ----------
        model :
            A ``pythtb`` model instance.
        spinful :
            Whether the model is spinful or not.
        point :
            A point in the reciprocal space.
        dim :
            Dimensionality of the reciprocal space.

    Returns
    -------
        hamilton :
            Hamiltonian matrix calculated in ``point``.
    """
    try:
        # Raises AttributeError if the model is not pythtb.TBModel
        if dim == model.dim_k:
            ham = model.hamiltonian(point, flatten_spin_axis=True)
        else:
            ham = model.hamiltonian(flatten_spin_axis=True)
    except AttributeError:
        if dim == model._dim_k:
            ham = model._gen_ham(point)
        else:
            ham = model._gen_ham()
        occ = model.get_num_orbitals() if spinful else model.get_num_orbitals() // 2
        ham = ham if not spinful else ham.reshape((2 * occ, 2 * occ))

    return ham.squeeze()


def get_half_filling(model, spinful):
    r"""Returns the number of occupied states at half-filling.

    Parameters
    ----------
        model :
            A ``pythtb`` model instance.
        spinful :
            Whether the model is spinful or not.

    Returns
    -------
        nocc :
            Number of occupied states at half-filling.
    """
    try:
        # Raises AttributeError if the model is not pythtb.TBModel
        return model.norb if spinful else model.norb // 2
    except AttributeError:
        return model.get_num_orbitals() if spinful else model.get_num_orbitals() // 2


def calc_states_uc(model, spinful):
    r"""Returns the number of states per unit cell.

    Parameters
    ----------
        model :
            A ``pythtb`` model instance.
        spinful :
            Whether the model is spinful or not (needed to properly account for spinful models).

    Returns
    -------
        size :
            Number of states per unit cell in the model.
    """
    try:
        # Raises AttributeError if the model is not pythtb.TBModel
        return model.norb * (2 if spinful else 1)
    except AttributeError:
        return model.get_num_orbitals() * (2 if spinful else 1)


def initialize_mask(model, spinful):
    r"""Returns a list of True for each state of the model.

    Parameters
    ----------
        model :
            A ``pythtb`` model instance.
        spinful :
            Whether the model is spinful or not (needed to properly account for spinful
            models).

    Returns
    -------
        mask :
            A list of :python:`True` values with the same dimension of the total number
            of orbitals in the model.
    """
    try:
        # Raises AttributeError if the model is not pythtb.TBModel
        return np.array([True for _ in range(model.norb * (2 if spinful else 1))])
    except AttributeError:
        return np.array(
            [True for _ in range(model.get_num_orbitals() * (2 if spinful else 1))]
        )


def calc_uc_vol(model):
    r"""Returns the volume of a 2D unit cell.

    Parameters
    ----------
        model :
            A ``pythtb`` model instance.

    Returns
    -------
        vol_uc :
            Volume of the 2D unit cell of the model.
    """
    try:
        # Raises AttributeError if the model is not pythtb.TBModel
        lat = model.get_lat_vecs()
    except AttributeError:
        lat = model.get_lat()

    return np.linalg.norm(np.cross(lat[0], lat[1]))


def make_finite(model, lx, ly):
    r"""Returns an instance of a model with every periodic hopping removed (a
    finite model within open boundary conditions).

    Parameters
    ----------
        model :
            A ``pythtb`` model instance.
        lx :
            Number of unit cells of the sample along the :math:`\mathbf{a}_1` direction.
        ly :
            Number of unit cells of the sample along the :math:`\mathbf{a}_2` direction.

    Returns
    -------
        finite :
            A model whose periodic hoppings have been removed (OBC model).
    """
    if not (lx > 0 and ly > 0):
        raise RuntimeError("Number of sites along finite direction must be positive.")

    try:
        # Raises AttributeError if the model is not pythtb.TBModel
        ribbon = model.cut_piece(num_cells=ly, periodic_dir=1, glue_edges=False)
        finite = ribbon.cut_piece(num_cells=lx, periodic_dir=0, glue_edges=False)
    except (AttributeError, TypeError):
        ribbon = model.cut_piece(num=ly, fin_dir=1, glue_edgs=False)
        finite = ribbon.cut_piece(num=lx, fin_dir=0, glue_edgs=False)

    return finite
