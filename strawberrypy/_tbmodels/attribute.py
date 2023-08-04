import numpy as np
import tbmodels as tbm

def _reciprocal_vec(model):
    """
    Returns the cartesian coordinates of the reciprocal lattice vectors
    """
    b_matrix = model.reciprocal_lattice
    b1 = b_matrix[0,:]
    b2 = b_matrix[1,:]
    return b1, b2

def get_positions(model, nx_sites = 1, ny_sites = 1):
    """
    Returns the cartesian coordinates of the orbitals centers states for a tbmodels.Model
    """
    positions = np.copy(model.pos)
    for i in range(model.pos.shape[0]):
        positions[i][0] *= nx_sites
        positions[i][1] *= ny_sites
        cartesian_pos = np.dot(positions[i], model.uc)
        positions[i] = cartesian_pos

    return positions

def _orb_cart(model):
    #returns position of orbitals in cartesian coordinates
    orb_red = model.pos
    n_orb = len(model.pos)
    lat_super = model.uc

    orb_c = []
    for i in range (n_orb):
        orb_c.append((np.matmul(lat_super.transpose(),orb_red[i].reshape(-1,1))).squeeze())
    orb_c = np.array(orb_c)
    return orb_c

def get_hamiltonian(model, point):
    """
    Returns the hamiltonian at the given k point and the number of occupied states for a tbmodels.Model
    """

    return model.hamilton(point, convention = 1), model.occ

def calc_states_uc(model):
    """
    Returns the number of states per unit cell of a tbmodels.Model
    """
    return model.size

def initialize_mask(model):
    """
    Returns a list of True for each state of the model
    """
    return np.array([True for _ in range(model.size)])

def calc_uc_vol(model):
    """
    Returns the volume of a 2D unit cell
    """
    return np.linalg.norm(np.cross(model.uc[0], model.uc[1]))

def cut_piece_tbm(source_model, num : int, fin_dir : int, dimk : int, glue : bool = False):
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

def make_finite(model, lx, ly):
    """
    Returns an instance of a tbmodels.Model with OBCs
    """

    if not (lx > 0 and ly > 0):
        raise RuntimeError("Number of sites along finite direction must be greater than 0")

    ribbon = cut_piece_tbm(model, num = ly, fin_dir = 1, dimk = 1, glue = False)
    finite = cut_piece_tbm(ribbon, num = lx, fin_dir = 0, dimk = 0, glue = False)

    return finite