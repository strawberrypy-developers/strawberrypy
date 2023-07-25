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

        if spinful == None:
            raise NotImplementedError("Please, specify if the model is spinful or not!")
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
            return _tbmodels.get_positions(self.model, self.Lx, self.Ly)          #Ho tolto il T perche' a me le posizioni servono cosi'
        elif isinstance(self.model, ptb.tb_model):
            return _pythtb.get_positions(self.model, self.spinful)
        else:
            raise NotImplementedError("Invalid model instance.")

    
    def _get_hamiltonian(self):
        """
        Returns the hamiltonian matrix at Gamma point
        """
        if isinstance(self.model, tbm.Model):
            gamma_point = np.zeros(self.dim)
            return _tbmodels.get_hamiltonian(self.model, gamma_point)
        elif isinstance(self.model, ptb.tb_model):
            gamma_point = np.zeros(self.dim)
            return _pythtb.get_hamiltonian(self.model, self.spinful, gamma_point, self.dim)
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
        Add vacancies in the systems by removing a site in the lattice.

        Args:
            - vacancies_list: a list of [cell_x, cell_y, basis] pointing to the cell and atom to be removed. Multiple values at once are allowed
        """
        ham = self.hamiltonian
        pos = self.r
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
                ham = np.delete(ham, vacancies, 0)
                ham = np.delete(ham, vacancies, 1)
                for dim in range(len(self.r)):
                    pos[dim] = np.delete(pos[dim], vacancies)
                    pos[dim] = np.delete(pos[dim], vacancies)
                if sz is not None:
                    sz = np.delete(sz, vacancies, 0)
                    sz = np.delete(sz, vacancies, 1)
                nocc -= 1
                norbs -= 1
            else:
                self._mask[vacancies + 1] = False
                for _ in range(2):
                    ham = np.delete(ham, vacancies, 0)
                    ham = np.delete(ham, vacancies, 1)
                    for dim in range(len(self.r)):
                        pos[dim] = np.delete(pos[dim], vacancies)
                        pos[dim] = np.delete(pos[dim], vacancies)
                    if sz is not None:
                        sz = np.delete(sz, vacancies, 0)
                        sz = np.delete(sz, vacancies, 1)
                nocc -= 2
                norbs -= 2
        else:
            vacancies = np.sort(vacancies)[::-1]
            for h in vacancies:
                self._mask[h] = False
                if not self.spinful:
                    ham = np.delete(ham, h, 0)
                    ham = np.delete(ham, h, 1)
                    for dim in range(len(self.r)):
                        pos[dim] = np.delete(pos[dim], h)
                        pos[dim] = np.delete(pos[dim], h)
                    if sz is not None:
                        sz = np.delete(sz, h, 0)
                        sz = np.delete(sz, h, 1)
                    nocc -= 1
                    norbs -= 1
                else:
                    self._mask[h] = False
                    self._mask[h + 1] = False
                    for _ in range(2):
                        ham = np.delete(ham, h, 0)
                        ham = np.delete(ham, h, 1)
                        for dim in range(len(self.r)):
                            pos[dim] = np.delete(pos[dim], h)
                            pos[dim] = np.delete(pos[dim], h)
                        if sz is not None:
                            sz = np.delete(sz, h, 0)
                            sz = np.delete(sz, h, 1)
                    nocc -= 2
                    norbs -= 2
        
        self.hamiltonian = ham
        self.sz = sz
        self.n_occ = nocc
        self.disordered = True

    #################################################
    # Utility functions
    #################################################

    def _xy_to_index(self, cellx : int, celly : int, basis : int):
        """
        Convert [cell_x, cell_y, basis] to the internal indexing of the lattice sites
        """
        return self.Ly * self.atoms_uc * cellx + self.states_uc * celly + basis * (2 if self.spinful else 1)
    

