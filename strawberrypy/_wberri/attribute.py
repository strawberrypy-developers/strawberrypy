import numpy as np
from wannierberri import Grid
from wannierberri.data_K import Data_K_R


def _reciprocal_vec(model):
    """
    Returns reciprocal lattice vectors in cartesian coordinates. ``wannierberri.System_w90`` version.

    Parameters
    ----------
        model :
            A ``wannierberri.System_w90`` instance.

    Returns
    -------
        b1, b2 :
            Reciprocal lattice vectors.
    """
    b_matrix = model.recip_lattice
    b1 = b_matrix[0,:]
    b2 = b_matrix[1,:]
    return b1, b2

def get_positions(model):
    """
    Returns the cartesian coordinates of the centers of Wannier functions. ``wannierberri.System_w90`` version.

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

def get_hamiltonian(model):
    r"""
    Returns the Wannier Hamiltonian at the :math:`\Gamma`-point (see Eq. (13) in Ref. `Marrazzo et al. (2023) <https://arxiv.org/abs/2312.10769)>`_ ) and the Data_K_R object containing information on the FFT grid for a ``wannierberri.System_w90`` instance defined by R-space matrices.

    Parameters
    ----------
        model :
            A ``wannierberri.System_w90`` instance.

    Returns
    -------
        ham :
            Interpolated Hamiltonian matrix in the Wannier gauge calculated at the :math:`\Gamma`-point.
        data :
            ``wannierberri.System_w90`` object for extracting k-space Wannier interpolated matrices.
    """

    print('Reading Hamiltonian at Gamma point in Wannier gauge..')
    grid = Grid(model, NK=model.NKFFT_recommended)
    dK = 1. / grid.div
    data = Data_K_R(model, dK, grid)

    Ham_W_R = model.Ham_R.copy()
    Ham_W_k = data.fft_R_to_k(Ham_W_R, hermitean=True)
    ham = Ham_W_k[0,:,:]

    return ham, data

def read_spn(model, data, u_n0): 
    r"""
    Returns the Wannier interpolated spin matrix :math:`S^(H)_z` (see Eq. (25) in Ref. `Marrazzo et al. (2023) <https://arxiv.org/abs/2312.10769)>`_ ) at the :math:`\Gamma`-point if seedname.spn file is provided.

    Parameters
    ----------
        model :
            A ``wannierberri.System_w90`` instance.
        data :
            ``wannierberri.System_w90`` object for extracting k-space Wannier interpolated matrices.
        u_n0 :
            Matrix of Hamiltonian eigenstates at :math:`\Gamma`-point (unitary matrix :math:`\mathcal{U}` in Eq. (25) in Ref. `Marrazzo et al. (2023) <https://arxiv.org/abs/2312.10769)>`_ ).

    Returns
    -------
        Sz :
            Wannier interpolated spin matrix calculated at the :math:`\Gamma`-point.
    """

    print('Reading Spin matrix at Gamma point in Wannier gauge..')

    SS = data.fft_R_to_k(model.get_R_mat('SS').copy(), hermitean=True)
    Sz_W = SS[0,:,:,2]
    Sz = np.conj(u_n0) @ Sz_W @ u_n0.T
    
    return Sz

def calc_states_uc(model):
    """
    Returns the number of Wannier functions per unit cell for a wannierberri.System_w90
    """
    return model.num_wann

def initialize_mask(model):
    """
    Returns a list of True for each state of the model
    """
    return np.array([True for _ in range(model.num_wann)])

