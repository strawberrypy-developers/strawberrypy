import numpy as np

import tbmodels as tbm
import pythtb as ptb

from . import _tbmodels
from . import _pythtb

class Model():
    def __init__(self, tbmodel = None, spinful : bool = False, states_uc : int = None, dim : int = None, Lx : int = 1, Ly : int = 1):
        
        # Store local variables
        self.model = tbmodel
        self.Lx = Lx
        self.Ly = Ly
        self.spinful = spinful

        # Generate the positions of the atoms in the lattice
        self.cart_positions = self._get_positions() 
        self.n_orb = self.cart_positions[:,0].size

        self.dim = self._get_dim(tbmodel) if dim == None else dim
        self.hamiltonian, self.n_occ = self._get_hamiltonian()

        # Generate the spin matrices if the model is spinful
        if spinful:
            self.sz = self._get_spin()
        else:
            self.sz = None

        # Generate the position matrices 
        self.r = []
        for d in range (self.dim):
            self.r.append( np.diag(self.cart_positions[:,d]) )

        # Number of states per unit cell
        if states_uc == None:
            self.states_uc = self._calc_states_uc()
        else:
            self.states_uc = states_uc
        
        # Disordered system
        self.disordered = False
        self._mask = self._initialize_mask()

    #################################################
    # Load functions
    #################################################

    def _get_positions(self):
        """
        Returns the cartesian coordinates of the states in a finite sample
        """
        if isinstance(self.model, tbm.Model):
            return _tbmodels.get_positions(self.model)
        elif isinstance(self.model, ptb.tb_model):
            return _pythtb.get_positions(self.model, self.spinful)
        else:
            raise NotImplementedError("Invalid model instance.")

    
    def _get_hamiltonian(self, external_model = None):
        """
        Returns the hamiltonian matrix at Gamma point
        """
        model_use = self.model if external_model is None else external_model
        if isinstance(model_use, tbm.Model):
            gamma_point = np.zeros(self.dim)
            return _tbmodels.get_hamiltonian(model_use, gamma_point)
        elif isinstance(model_use, ptb.tb_model):
            gamma_point = np.zeros(self.dim)
            return _pythtb.get_hamiltonian(model_use, self.spinful, gamma_point, self.dim)
        else:
            raise NotImplementedError("Invalid model instance.")
        

    def _get_spin(self):
        """
        Returns the spin matrix elements in the basis of TB orbitals which are diagonal in the spin z
        """
        if isinstance(self.model, tbm.Model) or isinstance(self.model, ptb.tb_model):
            return np.diag([1, -1] * (self.n_orb//2))
        else:
            raise NotImplementedError("Invalid model instance.")
        

    def _get_dim(self, model):
        """
        Returns the dimensionaloty of the model
        """
        if isinstance(model, tbm.Model):
            return model.dim
        elif isinstance(model, ptb.tb_model):
            return model._dim_k
        else:
            raise NotImplementedError("Invalid model instance.")
        

    def _calc_states_uc(self, model, spinful):
        """
        Returns number of states per unit cell
        """
        if isinstance(model, tbm.Model):
            return _tbmodels.calc_states_uc(model)
        elif isinstance(model, ptb.tb_model):
            return _pythtb.calc_states_uc(model, spinful)
        else:
            return NotImplementedError("Invalid model instance.")
        
        
    def _initialize_mask(self):
        """
        Returns a list of True values with the dimension of the lattice
        """
        if isinstance(self.model, tbm.Model):
            return _tbmodels.initialize_mask(self.model)
        elif isinstance(self.model, ptb.tb_model):
            return _pythtb.initialize_mask(self.model, self.spinful)
        else:
            return NotImplementedError("Invalid model instance.")

    #################################################
    # Lattice functions
    #################################################

    def add_onsite_disorder(self, w : float = 0, seed : int = None):
        """
        Add onsite (Anderson) disorder to the specified model. The disorder amplitude per site is taken randomly in [-w/2, w/2].

        Args:
            - w : disorder amplitude
            - seed : seed for random number generator
        """
        if w != 0: self.disordered = True

        # Set the seed for the random number generator
        if seed is not None:
            np.random.seed(seed)

        if self.spinful: 
            spinstates = 2
        else:
            spinstates = 1

        d = 0.5*w*(2*np.random.rand(self.n_orb//spinstates)-1.0)                   #create an array of random numbers in (-w/2,w/2)
        dis = np.repeat(d,spinstates)

        np.fill_diagonal(self.hamiltonian,self.hamiltonian.diagonal()+dis)         #function returns None, modifies self.hamiltonian


    def add_vacancies(self, vacancies_list):
        """
        Add vacancies in the systems by removing a site in the lattice. In order to enforce the half-filling, if the model is not
        spinful, an even number ov vacancies is required.

        Args:
            - vacancies_list: a list of [cell_x, cell_y, basis] pointing to the cell and atom to be removed. Multiple values at once are allowed
        """
        ham = self.hamiltonian
        pos = self.r
        cart_pos = self.cart_positions.T
        sz = self.sz
        nocc = self.n_occ
        norbs = self.n_orb

        # Convert vacancies_list to internal indexing
        if np.array(vacancies_list).shape == (3,):
            vacancies = self._xy_to_index(vacancies_list[0], vacancies_list[1], vacancies_list[2])
        else:
            vacancies = []
            for hl in vacancies_list:
                vacancies.append(self._xy_to_index(hl[0], hl[1], hl[2]))

        # Remove the selected sites
        if np.array(vacancies).shape == ():
            self._mask[vacancies] = False
            if not self.spinful:
                raise RuntimeError("Adding an odd number of vacancies causes the model to no longer be half-filled")
            else:
                self._mask[vacancies + 1] = False
                for _ in range(2):
                    ham = np.delete(ham, vacancies, 0)
                    ham = np.delete(ham, vacancies, 1)
                    for dim in range(len(self.r)):
                        pos[dim] = np.delete(pos[dim], vacancies, 0)
                        pos[dim] = np.delete(pos[dim], vacancies, 1)
                    cart_pos = np.delete(cart_pos, vacancies, 1)
                    sz = np.delete(sz, vacancies, 0)
                    sz = np.delete(sz, vacancies, 1)
                nocc -= 1
                norbs -= 2
        else:
            # Avoid relabelling
            vacancies = np.sort(vacancies)[::-1]

            # Check if the number of vacancies is odd if the model is not spinful
            if (not self.spinful) and (len(vacancies) % 2 == 1):
                raise RuntimeError("Adding an odd number of vacancies causes the model to no longer be half-filled")

            for h in vacancies:
                self._mask[h] = False
                if not self.spinful:
                    ham = np.delete(ham, h, 0)
                    ham = np.delete(ham, h, 1)
                    for dim in range(len(self.r)):
                        pos[dim] = np.delete(pos[dim], h, 0)
                        pos[dim] = np.delete(pos[dim], h, 1)
                    cart_pos = np.delete(cart_pos, h, 1)
                    nocc -= 0.5
                    norbs -= 1
                else:
                    self._mask[h] = False
                    self._mask[h + 1] = False
                    for _ in range(2):
                        ham = np.delete(ham, h, 0)
                        ham = np.delete(ham, h, 1)
                        for dim in range(len(self.r)):
                            pos[dim] = np.delete(pos[dim], h, 0)
                            pos[dim] = np.delete(pos[dim], h, 1)
                        cart_pos = np.delete(cart_pos, h, 1)
                        sz = np.delete(sz, h, 0)
                        sz = np.delete(sz, h, 1)
                    nocc -= 1
                    norbs -= 2
        
        self.hamiltonian = ham
        self.sz = sz
        self.n_occ = int(nocc)
        self.disordered = True
        self.cart_positions = cart_pos.T


    def make_heterostructure(self, model2, direction : int = 0, start : int = 0, stop : int = 0):
        """
        Modify a FiniteModel by merging another model in it. The system will be split in the direction starting from start.
        Beware: the previous model will be modified

        Args:
            - model2: the model that has to be merged into the existing one
            - direction : direction in which the splitting happen, allowed 0 for 'x' or 1 for 'y'
            - start : starting point for the splitting in the 'direction' direction
            - end : end point of the splitting in the 'direction' direction
        """

        # Check input data are ok
        if not start < stop:
            raise RuntimeError("Starting point is greater or equal to the end point")
        if not (start >= 0 and start < (self.Lx if direction == 0 else self.Ly)):
            raise RuntimeError("Start point value not allowed")
        if not (stop > 0 and stop < (self.Lx if direction == 0 else self.Ly)):
            raise RuntimeError("End point value not allowed")
        if direction not in [0, 1]:
            raise RuntimeError("Direction not allowed: insert 0 for 'x' and 1 for 'y'")
        
        if not issubclass(type(model2), Model):
            raise RuntimeError("The two models must be instances of Model, Supercell or FiniteModel")
        
        if not (self.Lx == model2.Lx and self.Ly == model2.Ly and np.allclose(self.r, model2.r)):
            raise RuntimeError("You can only build heterostructures of the same model")
        
        # Generate the Hamiltonian of the second model
        hamilt_model2 = np.copy(model2.hamiltonian)

        # Check if only the onsite terms are changed
        onsite_only = False
        if np.allclose( self.hamiltonian - np.diag(np.diag(self.hamiltonian)), hamilt_model2 - np.diag(np.diag(hamilt_model2)) ):
            onsite_only = True

        # Remove onsite terms from the model
        onsite1 = np.diag(self.hamiltonian).copy()
        onsite2 = np.diag(hamilt_model2).copy()
        self.hamiltonian -= np.diag(onsite1)
        
        if direction == 0:
            # Splitting along the x direction
            ind = np.array([[(start + i) * self.Ly * self.states_uc + j * self.states_uc for j in range(self.Ly)] for i in range(stop - start + 1)]).flatten()
        else:
            # Splitting along the y direction
            ind = np.array([[i * self.Ly * self.states_uc + start * self.states_uc + j * self.states_uc for j in range(stop - start + 1)] for i in range(self.Lx)]).flatten()

        for i in ind:
            for j in range(self.states_uc):
                onsite1[i + j] = onsite2[i + j]

        # Add the new onsite terms
        self.hamiltonian += np.diag(onsite1)
        
        # If other matrix element are changed
        if not onsite_only:
            # Indices of every atom in the selected cells, not only of the initial atom of the cell
            indices = np.array([[i + j for j in range(self.states_uc)] for i in ind]).flatten()

            # Cycle over the rows of the hopping matrix
            for k in range(self.hamiltonian.shape[0]):

                # Cycle over the columns of the hopping matrix
                for l in range(self.hamiltonian.shape[1]):
                    if k == l: continue

                    # Hopping amplitudes
                    amplitude1 = self.hamiltonian[k][l]
                    amplitude2 = hamilt_model2[k][l]
                            
                    if k in indices:
                        if np.absolute(amplitude2) < 1e-10: continue
                        self.hamiltonian[k][l] = amplitude2
                    else:
                        if np.absolute(amplitude1) < 1e-10: continue
                        self.hamiltonian[k][l] = amplitude1

    #################################################
    # Utility functions
    #################################################

    def _xy_to_index(self, cellx : int, celly : int, basis : int):
        """
        Convert [cell_x, cell_y, basis] to the internal indexing of the lattice sites
        """
        return self.Ly * self.states_uc * cellx + self.states_uc * celly + basis * (2 if self.spinful else 1)


    def _xy_to_line(self, fixed_coordinate : str, xy : int):
        """
        Returns the indices of the sites of the lattice keeping fixed the direction fixed_coordinate starting from xy
        """
        if fixed_coordinate == 'x':
            return np.array([self.states_uc * xy * self.Ly + i for i in range(self.states_uc * self.Ly) if self._mask[self.states_uc * xy * self.Ly + i]]).flatten().tolist()
        elif fixed_coordinate == 'y':
            indices = []
            for i in range(self.Lx):
                for j in range(self.states_uc):
                    if self._mask[self.states_uc * xy + self.states_uc * self.Ly * i + j]:
                        indices.append(self.states_uc * xy + self.states_uc * self.Ly * i + j)
            return np.array(indices)
        else:
            raise RuntimeError("Direction not allowed, only 'x' or 'y'")

    #################################################
    # Macroscopic average functions
    #################################################

    def _lattice_contraction(self, cutoff : float):
        """
        Defines which atomic sites must be contracted on one site, for each site of the lattice
        """
        contraction = []

        rx = self.cart_positions[:, 0]; ry = self.cart_positions[:, 1]
        def within_range(current, trial):
            return True if (rx[current] - rx[trial]) ** 2 + (ry[current] - ry[trial]) ** 2 - cutoff ** 2 < 1e-6 else False

        for current in range(len(rx)):
            contraction.append([trial for trial in range(len(rx)) if within_range(current, trial)])

        return contraction
    

    def _PBC_lattice_contraction(self, cutoff):
        """
        Defines which atomic sites must be contracted on one site, for each site of the lattice within PBC (minimum convention image)
        """
        contraction = []

        if isinstance(self.model, tbm.Model):
            lvecs = np.copy(self.model.uc)
        elif isinstance(self.model, ptb.tb_model):
            lvecs = np.copy(self.model._lat)
        else:
            raise NotImplementedError("Invalid model instance.")

        lvecs[0] /= self.Lx
        lvecs[1] /= self.Ly
        
        # Compute reduced coordinates
        reds = []
        linv = np.linalg.inv(lvecs)
        for r in range(len(self.cart_positions)):
            reds.append(np.dot(np.array([self.cart_positions[r, 0], self.cart_positions[r, 1]]), linv))
        reds = np.array(reds)
        rx = reds[:, 0]; ry = reds[:, 1]

        def within_range(current, trial):
            trialorbitals = np.array([
                [rx[trial], ry[trial]], [rx[trial], ry[trial] + self.Ly], [rx[trial], ry[trial] - self.Ly], [rx[trial] + self.Lx, ry[trial]],
                [rx[trial] + self.Lx, ry[trial] + self.Ly], [rx[trial] + self.Lx, ry[trial] - self.Ly], [rx[trial] - self.Lx, ry[trial]],
                [rx[trial] - self.Lx, ry[trial] + self.Ly], [rx[trial] - self.Lx, ry[trial] - self.Ly]
            ])
            for to in trialorbitals:
                dr = np.dot(np.array([rx[current] - to[0], ry[current] - to[1]]), lvecs)
                if np.dot(dr, dr) - cutoff ** 2 < 1e-6:
                    return True
            return False

        for current in range(len(rx)):
            contraction.append([trial for trial in range(len(rx)) if within_range(current, trial)])

        return contraction


    def _average_over_radius(self, vals, cutoff : float, contraction = None):
        """
        Average vals over the contraction of the lattice defined by the cutoff radius
        """
        return_vals = []
        if contraction is None:
            contraction = self._lattice_contraction(cutoff)

        # Macroscopic average within a certain radius
        for current in range(len(self.cart_positions[:, 0])):
            tmp = [vals[ind] for ind in contraction[current]]
            if not len(tmp) == 0:
                return_vals.append(np.sum(tmp) / (len(tmp) / self.states_uc))
            else:
                raise RuntimeError("Unexpected error occourred in counting the neighbors of a lattice site, there are none")

        return np.array(return_vals)