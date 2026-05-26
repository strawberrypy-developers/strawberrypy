import numpy as np
import wannierberri as wberri

wberri_version = wberri.__version__


def _reciprocal_vec(model):
    r"""Returns reciprocal lattice vectors in cartesian coordinates.

    Parameters
    ----------
        model :
            A ``wannierberri.System_w90`` instance.

    Returns
    -------
        b1, b2 :
            Reciprocal lattice vectors.
    """
    return model.recip_lattice


def get_positions(model):
    r"""Returns the cartesian coordinates of the centers of Wannier functions.

    Parameters
    ----------
        model :
            A ``wannierberri.System_w90`` instance.

    Returns
    -------
        positions :
            Cartesian coordinates of the centers of Wannier functions.
    """
    return model.wannier_centers_cart


def get_hamiltonian(model, mp_grid: list[int, int, int] = [1, 1, 1], **kwargs):
    r"""Returns the Wannier Hamiltonian at the :math:`\Gamma`-point (see Eq. 13) in Ref.
    `Marrazzo et al. (2023) <https://arxiv.org/abs/2312.10769)>`_ ) and the Data_K_R object
    containing information on the FFT grid for a ``wannierberri.System_w90`` instance
    defined by R-space matrices.

    Parameters
    ----------
        model :
            A ``wannierberri.System_w90`` instance.

    Returns
    -------
        ham :
            Interpolated Hamiltonian matrix in the Wannier gauge calculated at the
            :math:`\Gamma`-point.
        mp_grid :
            MP grid dimensions used in the .WIN file for the WannierBerri model.
    """
    grid = wberri.Grid(model, NK=mp_grid)
    data = wberri.data_K.Data_K_R(model, [0, 0, 0], grid)
    Ham_W_k = data.rvec.R_to_k(model.Ham_R.copy(), hermitian=True)

    return Ham_W_k[0, :, :], data


def read_spn(model, data):
    r"""Returns the Wannier interpolated spin matrix :math:`S^(W)_z` (see Eq. (25) in Ref.
    `Marrazzo et al. (2023) <https://arxiv.org/abs/2312.10769)>`_ ) at the :math:`\Gamma`-point
    if seedname.spn file is provided.

    Parameters
    ----------
        model :
            A ``wannierberri.System_w90`` instance.
        data :
            ``wannierberri.System_w90`` object for extracting k-space Wannier interpolated
            matrices.

    Returns
    -------
        Sz :
            Wannier interpolated spin matrix calculated at the :math:`\Gamma`-point.
    """
    SS = data.rvec.R_to_k(model.get_R_mat("SS").copy(), hermitian=True)

    return SS[0, :, :, 2]


def calc_states_uc(model):
    r"""Returns the number of Wannier functions per unit cell for a wannierberri.System_w90."""
    return model.num_wann


def initialize_mask(model):
    r"""Returns a list of True for each state of the model."""
    return np.array([True for _ in range(model.num_wann)])


def get_model_wberri(path_seedname: str = None, spin: bool = False) -> wberri.System_w90:
    r"""Create a ``wannierberri.System_w90`` instance from the Wannier90
    seedname and path to the Wannier90 output files."""
    return wberri.System_w90(seedname=path_seedname, spin=spin)


def read_spn_from_seedname(
    path_seedname: str = None,
    spin: bool = False,
    mp_grid: list[int, int, int] = [1, 1, 1],
    **kwargs,
) -> np.ndarray:
    r"""Read spin matrices from a .SPN file using WannierBerri.

    Parameters
    ----------
        path_seedname : str
            Path to the Wannier90 seedname (without extension).
        spin : bool
            Whether to read the spin matrix as spinful (True) or spinless (False).
        mp_grid : list of int
            MP grid dimensions used in the .WIN file.
        **kwargs : dict
            Additional keyword arguments passed to WannierBerri's System_w90.

    Returns
    -------
        spin_matrices : np.ndarray
            Spin matrices in the Wannier gauge at Gamma point.
    """
    return_model = kwargs.pop("return_model", False)
    model_wb = get_model_wberri(path_seedname=path_seedname, spin=spin)
    data_wb = wberri.data_K.Data_K_R(
        model_wb, [0, 0, 0], wberri.Grid(model_wb, NK=mp_grid)
    )

    # Spin matrices in the Wannier gauge in Gamma
    spinmats = data_wb.rvec.R_to_k(model_wb.get_R_mat("SS").copy(), hermitian=True)

    if return_model:
        return model_wb, spinmats
    else:
        return spinmats


def calc_uc_vol(model):
    r"""Returns the unit cell volume for a ``wannierberri.System_w90`` instance."""
    uc = model.real_lattice
    return np.linalg.norm(np.cross(uc[0], uc[1]))
