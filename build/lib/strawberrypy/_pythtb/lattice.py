import numpy as np
import pythtb

def reciprocal_vec(model):
    #returns reciprocal lattice vectors
    lat = model.get_lat()     
    a_matrix = np.array([[lat[1,1], -lat[0,1]],[-lat[1,0], lat[0,0]]])
    b_matrix = (2.*np.pi / (lat[0,0]*lat[1,1]-lat[0,1]*lat[1,0])) * a_matrix

    b1 = b_matrix[:,0].reshape(-1,1)
    b2 = b_matrix[:,1].reshape(-1,1)
    return b1, b2

def orb_cart (model):
    #returns position of orbitals in cartesian coordinates
    n_orb = model.get_num_orbitals()
    lat_super = model.get_lat()          
    orb_red = model.get_orb()            

    orb_c = []
    for i in range (n_orb):
        orb_c.append( (np.matmul(lat_super.transpose(),orb_red[i].reshape(-1,1))).squeeze() )   
    orb_c = np.array(orb_c)
    return orb_c

def orb_cart_spin (model):
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

def periodic_gauge (u_n0, b, model):
    #returns the matrix of occupied eigenvectors at the edge of the Brillouin zone imposing periodic gauge
    n_orb = model.get_num_orbitals()
    n_occ = n_orb//2

    orb_c = orb_cart(model)
    vec_scal_b = orb_c @ b
    vec_exp_b = np.exp(-1.j*vec_scal_b)

    u_nb = vec_exp_b.T * u_n0[:n_occ,:]

    return u_nb

def periodic_gauge_spin(u_n0, b, model):
    #returns the matrix of occupied eigenvectors at the edge of the Brillouin zone imposing periodic gauge
    n_orb = model.get_num_orbitals()
    n_occ = n_orb

    orb_c = orb_cart_spin(model)
    vec_scal_b = orb_c @ b
    vec_exp_b = np.exp(-1.j*vec_scal_b)

    u_nb = vec_exp_b.T * u_n0[:n_occ,:]

    return u_nb

def onsite_disorder(source_model, w : float, spinstates : int = 2, seed : int = None):
    """
    Add onsite (Anderson) disorder to the specified model. The disorder amplitude per site is taken randomly in [-w/2, w/2].

        Args:
        - source_model : the model to add disorder to
        - w : disorder amplitude
        - spinstates : spin of the model
        - seed : seed for random number generator

        Returns:
        - model : the disordered model
    """
    # Quick return for no disorder
    if w == 0:
        return source_model

    if seed is not None:
        np.random.seed(seed)

    # Number of orbitals in the supercell model = norbs (original) x num
    norbs = source_model.get_num_orbitals()
    
    # Onsite energies per unit cell (2 is by convention with TBModels)
    disorder = 0.5 * w * (2 * np.random.rand(norbs // spinstates) - 1.0)
    disorder = np.repeat(disorder, spinstates)
    onsite = source_model._site_energies + disorder

    newmodel = pythtb.tb_model(dim_k = 0, dim_r = source_model._dim_r, lat = source_model._lat, orb = source_model._orb, nspin = source_model._nspin)
    newmodel.set_onsite(onsite, mode = 'set')

    # Cycle over the rows of the hopping matrix
    hoppings = source_model._hoppings
    for k in range(len(hoppings)):
        if np.absolute(hoppings[k][0]) < 1e-10: continue
        newmodel.set_hop(hoppings[k][0], hoppings[k][1], hoppings[k][2], mode = "add")
    
    return newmodel