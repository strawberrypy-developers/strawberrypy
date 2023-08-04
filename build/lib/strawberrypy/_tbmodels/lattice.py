import numpy as np
import tbmodels

def reciprocal_vec(model):
    #returns reciprocal lattice vectors
    b_matrix = model.reciprocal_lattice
    b1 = b_matrix[0,:]
    b2 = b_matrix[1,:]
    return b1, b2

def orb_cart(model):
    #returns position of orbitals in cartesian coordinates
    orb_red = model.pos
    n_orb = len(model.pos)
    lat_super = model.uc

    orb_c = []
    for i in range (n_orb):
        orb_c.append((np.matmul(lat_super.transpose(),orb_red[i].reshape(-1,1))).squeeze())
    orb_c = np.array(orb_c)
    return orb_c

def periodic_gauge(u_n0, b, model):
    #returns the matrix of occupied eigenvectors at the edge of the Brillouin zone imposing periodic gauge
    n_occ = model.occ

    orb_c = orb_cart(model)
    vec_scal_b = orb_c @ b
    vec_exp_b = np.exp(-1.j*vec_scal_b)

    u_nb = vec_exp_b.T * u_n0[:n_occ,:]

    return u_nb

def onsite_disorder(source_model, w : float = 0, spinstates : int = 2, seed : int = None):
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

    # Set the seed for the random number generator
    if seed is not None:
        np.random.seed(seed)

    # Number of orbitals in the supercell model = norbs (original) x num
    norbs = source_model.size
    
    # Onsite energies per unit cell (2 is by convention with TBModels)
    onsite = [2 * np.real(source_model.hop[source_model._zero_vec][j][j]) for j in range(norbs)]
    disorder = 0.5 * w * (2 * np.random.rand(norbs // spinstates) - 1.0)
    disorder = np.repeat(disorder, spinstates)
    onsite += disorder

    # Hopping amplitudes and positions
    hoppings = [[key, val] for key, val in iter(source_model.hop.items())]

    # Hoppings to be added
    hopping_list = []

    # Cycle over the number of defined hoppings
    for j in range(len(hoppings)):

        # Set lattice vector of the current hopping matrix
        objective = np.copy(hoppings[j][0])

        # Cycle over the rows of the hopping matrix
        for k in range(hoppings[j][1].shape[0]):

            # Cycle over the columns of the hopping matrix
            for l in range(hoppings[j][1].shape[1]):

                # Hopping amplitudes
                amplitude = hoppings[j][1][k][l]
                if np.absolute(amplitude) < 1e-10:
                    continue

                if k == l:
                    continue
                else:
                    hopping_list.append([amplitude, int(k), int(l), objective])
    
    model = tbmodels.Model.from_hop_list(hop_list = hopping_list, on_site = onsite, size = norbs, dim = source_model.dim,
        occ = source_model.occ, uc = source_model.uc, pos = source_model.pos, contains_cc = False)
    
    return model