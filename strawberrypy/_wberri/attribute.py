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
    Returns the Hamiltonian at the :math:`\Gamma`-point and Data_K_R object for reading k-space Wannier interpolated matrices. ``wannierberri.System_w9`` version.

    Parameters
    ----------
        model :
            A ``wannierberri.System_w9`` instance.

    Returns
    -------
        hamilton :
            Hamiltonian matrix calculated at the :math:`\Gamma`-point.
        data :
            ``wannierberri.System_w9`` object for reading k-space Wannier interpolated matrices.
    """

    print('Reading Hamiltonian at Gamma point in Wannier gauge..')
    grid = Grid(model, NK=model.NKFFT_recommended)
    dK = 1. / grid.div
    data = Data_K_R(model, dK, grid)

    ham_W_R = model.Ham_R.copy()
    Ham_W_k = data.fft_R_to_k(ham_W_R, hermitean=True)
    ham = Ham_W_k[0,:,:]

    return ham, data

def read_spn(model, data, u_n0): 
    r"""
    Returns the spin matrix elements in the basis of Wannier functions :math:`S^(H)_z` (see Eq. 25) if seedname.spn file is provided.

    Parameters
    ----------
        model :
            A ``wannierberri.System_w9`` instance.
        data :
            ``wannierberri.System_w9`` object for reading k-space Wannier interpolated matrices.
        u_n0 :
            Matrix of Hamiltonian eigenstates at :math:`\Gamma`-point for 

    Returns
    -------
        hamilton :
            Hamiltonian matrix calculated at the :math:`\Gamma`-point.
        data :
            ``wannierberri.System_w9`` object for reading k-space Wannier interpolated matrices.
    """

    print('Reading Spin matrix at Gamma point in Wannier gauge..')

    SS = data.fft_R_to_k(model.get_R_mat('SS').copy(), hermitean=True)
    sz_w = SS[0,:,:,2]
    sz = np.conj(u_n0) @ sz_w @ u_n0.T
    
    return sz

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

def calc_uc_vol(model):
    """
    Returns the volume of a 2D unit cell
    """
    return np.linalg.norm(np.cross(model._lat[0], model._lat[1]))

def make_finite(model, lx, ly):
    """
    Returns an instance of a pythtb.tb_model with OBCs
    """
    if not (lx > 0 and ly > 0):
        raise RuntimeError("Number of sites along finite direction must be greater than 0")

    ribbon = model.cut_piece(num = ly, fin_dir = 1, glue_edgs = False)
    finite = ribbon.cut_piece(num = lx, fin_dir = 0, glue_edgs = False)

    return finite