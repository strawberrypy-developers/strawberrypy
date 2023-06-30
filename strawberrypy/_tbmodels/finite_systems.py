import numpy as np
import tbmodels as tbm

def get_positions(model, nx_sites, ny_sites):
    """
    Returns the cartesian coordinates of the states in a finite sample of tbmodels.Model
    """
        
    positions = np.copy(model.pos)
    for i in range(model.pos.shape[0]):
        positions[i][0] *= nx_sites
        positions[i][1] *= ny_sites
        cartesian_pos = np.dot(positions[i], model.uc)
        positions[i] = cartesian_pos
    return positions

def _cut_piece_tbm(source_model, num : int, fin_dir : int, dimk : int, glue : bool = False):
    """
    Reduce by 1 the dimension of a model in TBModels in a given direction, building a supercell made by a given number of atoms. If a zero dimensional system, is needed it is required to run twice the function with dimk = 1 and then dimk = 0.
    
        Args:
        - source_model: the model to reduce
        - num: number of cell to keep in the finite direction
        - fin_dir: finite direction (ie, 0 = x, 1 = y)
        - dimk: number of periodic directions after the cut
        - glue: glue edges to make periodicity

        Returns:
        - tbmodels.Model finite along fin_dir
    """

    # Ensure the condition the function has been built for
    if num <= 0:
        raise RuntimeError("Negative number of cells in the finite direction required.")
    if fin_dir not in [0, 1]:
        raise RuntimeError("Finite direction not allowed (only 2D systems).")
    if dimk not in [0, 1]:
        raise RuntimeError("Leftover k-space dimension not allowed.")
    if num == 1 and glue == True:
        raise RuntimeError("Cannot glue edges with one cell in the finite direction.")
    
    # Number of orbitals in the supercell model = norbs (original) x num
    norbs = source_model.size

    # Define the supercell
    newpos = []
    for i in range(num):
        for j in range(norbs):
            # Convert coordinates into cartesian coordinates
            orb_tmp = np.copy(source_model.pos[j, :])

            # One direction is fine but I need to map the other into the unit cell
            orb_tmp[fin_dir] += float(i)
            orb_tmp[fin_dir] /= num

            # Convert the coordinates back into lattice coordinates
            newpos.append(orb_tmp)

    # Onsite energies per unit cell (2 is by convention with TBModels)
    onsite = num * [2 * np.real(source_model.hop[source_model._zero_vec][j][j]) for j in range(norbs)]

    # Hopping amplitudes and positions
    hoppings = [[key, val] for key, val in iter(source_model.hop.items())]

    # Hoppings to be added
    hopping_list = []

    # Cycle over the number of defined hoppings
    for j in range(len(hoppings)):

        # Set lattice vector of the current hopping matrix
        objective = np.copy(hoppings[j][0])

        # Maximum bond length
        jump_fin = hoppings[j][0][fin_dir]

        # If I have a finite direction I make the hopping vector finite, and
        #   if I have no periodic direction, I but every hopping in the [zero] cell
        if dimk != 0:
            objective[fin_dir] = 0
        else:
            objective = np.array([0 for i in range(source_model.dim)])

        # Cycle over the rows of the hopping matrix
        for k in range(hoppings[j][1].shape[0]):

            # Cycle over the columns of the hopping matrix
            for l in range(hoppings[j][1].shape[1]):

                # Hopping amplitudes
                amplitude = hoppings[j][1][k][l]
                if np.absolute(amplitude) < 1e-10:
                    continue

                # Cycle over the cells in the supercell
                for i in range(num):
                    starting = k + i * norbs
                    ending = l + (i + jump_fin) * norbs

                    # Decide wether to add the hopping or not
                    to_add = True

                    if not glue:
                        if ending < 0 or ending >= norbs * num:
                            to_add = False
                    else:
                        ending = int(ending) % int(norbs * num)

                    # Avoid setting twice onsite energies
                    if starting == ending and (objective == [0 for i in range(source_model.dim)]).all():
                        continue

                    if to_add == True:
                        hopping_list.append([amplitude, int(starting), int(ending), objective])
    
    model = tbm.Model.from_hop_list(hop_list = hopping_list, on_site = onsite, size = norbs * num, dim = source_model.dim,
        occ = source_model.occ * num, uc = source_model.uc, pos = newpos, contains_cc = False)
    
    return model

def make_finite(model: tbm.Model, nx_sites: int, ny_sites: int):
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

    ribbon = _cut_piece_tbm(model, num = ny_sites, fin_dir = 1, dimk = 1, glue = False)
    finite = _cut_piece_tbm(ribbon, num = nx_sites, fin_dir = 0, dimk = 0, glue = False)
    
    return finite

def make_heterostructure(model1 : tbm.Model , model2 : tbm.Model,  nx_sites : int, ny_sites : int, direction : int, start : int, stop : int):
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
    atoms_uc = int(model1.size / (nx_sites * ny_sites))

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
    if not (model1.size == model2.size and np.allclose(model1.pos, model2.pos) and np.allclose(model1.uc, model2.uc) and model1.occ == model2.occ):
        raise RuntimeError("The models to merge must be the same model with different parameters")

    # Make a copy of the model and remove all onsite terms
    onsite1 = np.diag(list(model1.hop.values())[0]).copy()
    onsite2 = np.diag(list(model2.hop.values())[0]).copy()
    
    if direction == 0:
        # Splitting along the x direction
        ind = np.array([[(start + i) * ny_sites * atoms_uc + j * atoms_uc for j in range(ny_sites)] for i in range(stop - start + 1)]).flatten()
    else:
        # Splitting along the y direction
        ind = np.array([[i * ny_sites * atoms_uc + start * atoms_uc + j * atoms_uc for j in range(stop - start + 1)] for i in range(nx_sites)]).flatten()

    for i in ind:
        for j in range(atoms_uc):
            onsite1[i + j] = onsite2[i + j]
    onsite1 = [2 * term for term in onsite1]

    # Indices of every atom in the selected cells, not only of the initial atom of the cell
    indices = np.array([[i + j for j in range(atoms_uc)] for i in ind]).flatten()

    # Hopping amplitudes and positions
    hoppings1 = [[key, val] for key, val in iter(model1.hop.items())][0][1]
    hoppings2 = [[key, val] for key, val in iter(model2.hop.items())][0][1]
    hopping_list = []

    # Cycle over the rows of the hopping matrix
    for k in range(hoppings1.shape[0]):

        # Cycle over the columns of the hopping matrix
        for l in range(hoppings1.shape[1]):
            if k == l: continue

            # Hopping amplitudes
            amplitude1 = hoppings1[k][l]
            amplitude2 = hoppings2[k][l]
                    
            if k in indices:
                if np.absolute(amplitude2) < 1e-10: continue
                hopping_list.append([amplitude2, k, l, [0 for i in range(model1.dim)]])
            else:
                if np.absolute(amplitude1) < 1e-10: continue
                hopping_list.append([amplitude1, k, l, [0 for i in range(model1.dim)]])

    newmodel = tbm.Model.from_hop_list(hop_list = hopping_list, on_site = onsite1, size = model1.size, dim = model1.dim,
        occ = model1.occ, uc = model1.uc, pos = model1.pos, contains_cc = False)

    return newmodel

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
    
    model = tbm.Model.from_hop_list(hop_list = hopping_list, on_site = onsite, size = norbs, dim = source_model.dim,
        occ = source_model.occ, uc = source_model.uc, pos = source_model.pos, contains_cc = False)
    
    return model
