import numpy as np

def _orb_cart (model):
    #returns position of orbitals in cartesian coordinates
    n_orb = model.get_num_orbitals()
    lat_super = model.get_lat()          
    orb_red = model.get_orb()            

    orb_c = []
    for i in range (n_orb):
        orb_c.append( (np.matmul(lat_super.transpose(),orb_red[i].reshape(-1,1))).squeeze() )   
    orb_c = np.array(orb_c)
    return orb_c

def _orb_cart_spin (model):
    #returns position of orbitals in cartesian coordinates
    n_occ = model.get_num_orbitals()
    lat_super = model.get_lat()          
    orb_red = model.get_orb()            

    orb_c = []
    for i in range (n_occ):
        orb_c.append( (np.matmul(lat_super.transpose(),orb_red[i].reshape(-1,1))).squeeze() )  
        orb_c.append( (np.matmul(lat_super.transpose(),orb_red[i].reshape(-1,1))).squeeze() )  
    orb_c = np.array(orb_c)
    return orb_c

def _reciprocal_vec(model):
    """
    Returns the cartesian coordinates of the reciprocal lattice vectors
    """
    lat = model.get_lat()     
    a_matrix = np.array([[lat[1,1], -lat[0,1]],[-lat[1,0], lat[0,0]]])
    b_matrix = (2.*np.pi / (lat[0,0]*lat[1,1]-lat[0,1]*lat[1,0])) * a_matrix

    b1 = b_matrix[:,0].reshape(-1,1)
    b2 = b_matrix[:,1].reshape(-1,1)
    return b1, b2

def get_positions(model, spinful):
    """
    Returns the cartesian coordinates of the orbitals centers states for a pythtb.tb_model
    """

    if not spinful:
        return _orb_cart(model)             
    else:
        tmp = _orb_cart(model).T
        newx = np.repeat(tmp[0], 2)
        newy = np.repeat(tmp[1], 2)
        return np.array([newx, newy]).T
    
def get_hamiltonian(model, spinful, point, dim):
    """
    Returns the hamiltonian at the given k point and the number of occupied states for a pythtb.tb_model
    """
        
    if dim == model._dim_k:
        ham = model._gen_ham(point)
    else:
        ham = model._gen_ham()
    occ = model.get_num_orbitals() if spinful else model.get_num_orbitals() // 2

    return ham if not spinful else ham.reshape((2 * occ, 2 * occ)), occ

def calc_states_uc(model, spinful):
    """
    Returns the number of states per unit cell for a pythtb.tb_model
    """
    return model.get_num_orbitals() * (2 if spinful else 1)

def initialize_mask(model, spinful):
    """
    Returns a list of True for each state of the model
    """
    return np.array([True for _ in range(model.get_num_orbitals() * (2 if spinful else 1))])

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