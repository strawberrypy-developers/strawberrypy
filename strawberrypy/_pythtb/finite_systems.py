import numpy as np
import pythtb as ptb

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

def make_finite(model, nx_sites: int, ny_sites: int):
    """
    Make a finite mdoel along x and y direction by first cutting on the y direction and then on the x. This convention has been used to track the positions in the functions

        Args:
        - model : instance of the model, which should be periodic in both x and y direction
        - nx_sites, ny_sites : number of sites of the finite sample in both directions

        Returns:
        - model : the finite model
    """

    if not (nx_sites > 0 and ny_sites > 0):
        raise RuntimeError("Number of sites along finite direction must be greater than 0")

    ribbon = model.cut_piece(num = ny_sites, fin_dir = 1, glue_edgs = False)
    finite = ribbon.cut_piece(num = nx_sites, fin_dir = 0, glue_edgs = False)
    
    return finite

def make_heterostructure(model1, model2,  nx_sites : int, ny_sites : int, direction : int, start : int, stop : int):
    """
    Modify a finite model by merging another system in it. The system will be split in the direction starting from start.

        Args:
        - model1, model2: the models which composes the heterostructure
        - nx_sites, ny_sites : x and y length of the finite system
        - direction : direction in which the splitting happen, allowed 0 for 'x' or 1 for 'y'
        - start : starting point for the splitting in the 'direction' direction
        - end : end point of the splitting in the 'direction' direction

        Returns:
        - model : the model composed my the two subsystems
    """

    # Number of states per unit cell
    atoms_uc = int(model1.get_num_orbitals() / (nx_sites * ny_sites))

    # Check validity of input data
    if not start < stop:
        raise RuntimeError("Starting point is greater or equal to the end point")
    if not (start >= 0 and start < (nx_sites if direction == 0 else ny_sites)):
        raise RuntimeError("Start point value not allowed")
    if not (stop > 0 and stop < (nx_sites if direction == 0 else ny_sites)):
        raise RuntimeError("End point value not allowed")
    if direction not in [0, 1]:
        raise RuntimeError("Direction not allowed: insert 0 for 'x' and 1 for 'y'")

    # Assert the model is the same
    if not (model1.get_num_orbitals() == model2.get_num_orbitals() and np.allclose(model1._orb, model2._orb) and np.allclose(model1._lat, model2._lat)):
        raise RuntimeError("The models to merge must be the same model with different parameters")

    # Make a copy of the model and remove all onsite terms
    onsite1 = np.copy(model1._site_energies)
    onsite2 = np.copy(model2._site_energies)
    
    if direction == 0:
        # Splitting along the x direction
        ind = np.array([[(start + i) * ny_sites * atoms_uc + j * atoms_uc for j in range(ny_sites)] for i in range(stop - start + 1)]).flatten()
    else:
        # Splitting along the y direction
        ind = np.array([[i * ny_sites * atoms_uc + start * atoms_uc + j * atoms_uc for j in range(stop - start + 1)] for i in range(nx_sites)]).flatten()

    for i in ind:
        for j in range(atoms_uc):
            onsite1[i + j] = onsite2[i + j]

    # Indices of every atom in the selected cells, not only of the initial atom of the cell
    indices = np.array([[i + j for j in range(atoms_uc)] for i in ind]).flatten()

    # Hopping amplitudes and positions
    hoppings1 = model1._hoppings; hoppings2 = model2._hoppings

    # The model to return
    newmodel = ptb.tb_model(dim_k = 0, dim_r = model1._dim_r, lat = model1._lat, orb = model1._orb, nspin = model1._nspin)
    newmodel.set_onsite(onsite1, mode = 'set')

    # Cycle over the rows of the hopping matrix
    for k in range(len(hoppings1)):
                    
        if k in indices:
            if np.absolute(hoppings2[k][0]) < 1e-10: continue
            newmodel.set_hop(hoppings2[k][0], hoppings2[k][1], hoppings2[k][2], mode = "add")
        else:
            if np.absolute(hoppings1[k][0]) < 1e-10: continue
            newmodel.set_hop(hoppings1[k][0], hoppings1[k][1], hoppings1[k][2], mode = "add")

    return newmodel

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

    newmodel = ptb.tb_model(dim_k = 0, dim_r = source_model._dim_r, lat = source_model._lat, orb = source_model._orb, nspin = source_model._nspin)
    newmodel.set_onsite(onsite, mode = 'set')

    # Cycle over the rows of the hopping matrix
    hoppings = source_model._hoppings
    for k in range(len(hoppings)):
        if np.absolute(hoppings[k][0]) < 1e-10: continue
        newmodel.set_hop(hoppings[k][0], hoppings[k][1], hoppings[k][2], mode = "add")
    
    return newmodel
