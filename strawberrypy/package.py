import numpy as np
import tbmodels as tbm
import pythtb as ptb
import scipy.linalg as la
from opt_einsum import contract

from . import _tbmodels
from . import _pythtb
from .common_functions import *

class FiniteModel:
    """
    A class describing a finite model with either TBmodels or PythTB instances. Ability to add disorder and vacancies to the models,
    calculate local topological markers and localization markers.
    The instance initialization can be performed in two ways:
    1) mode = 'tb': With this mode, FiniteModel expects to have as input a PBC instance of TBmodels or PythTB, the number of cells
        along x and y direction and a bool (spinful) which specifies if the model has spinful electrons;
    2) mode = 'load': With this mode, FiniteModel expects to have as input fnames, which is a list of filenames from which load the
        required quantities, namely [hamiltonian, x_coordinate, y_coordinate, sz_spin]. Moreover, is required the area of the unit
        cell, the number of states per unit cell, and the same spinful argument of the case above;
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, tbmodel = None, nx_sites : int = 1, ny_sites : int = 1, spinful : bool = False, mode : str = 'tb', fnames = None, uc_vol : float = 1, atoms_uc : int = 1):
        if not mode in ['load', 'tb']:
            raise RuntimeError("Mode not allowed, only 'load' from file or 'tb' to run a tight-binding model")
        
        # Store local variables
        self._nx_sites = nx_sites
        self._ny_sites = ny_sites
        self._mode = mode
        self._spinful = spinful

        # Creation of the finite model
        if mode == 'tb':
            self._model = make_finite(tbmodel, nx_sites, ny_sites)
            self._uc_vol = self._calc_uc_vol()
            self._atoms_uc = self._calc_atoms_uc()
        else:
            self._model = None
            self._uc_vol = uc_vol
            self._atoms_uc = atoms_uc

        # Generate the spin matrices if the model is spinful
        if spinful:
            if mode == 'tb':
                self._sz = self._get_spin()
            else:
                self._sz = self._load_spin("szspin.input" if not fnames == None else fnames[3])
        else:
            self._sz = None
        
        # Generate the positions of the atoms in the lattice
        if mode == 'tb':
            positions = self._get_positions()
        else:
            positions = self._load_positions(["xcoordinates.input"  if not fnames == None else fnames[1], "ycoordinates.input"  if not fnames == None else fnames[2]])
        self._rx = positions[0]
        self._ry = positions[1]

        # Generate the hamiltonian 
        if mode == 'tb':
            self._hamiltonian, self._nocc = self._get_hamiltonian()
        else:
            self._hamiltonian, self._nocc = self._load_hamiltonian("hamiltonian.input" if not fnames == None else fnames[0])

        # Misc
        self._disordered = False
        self._mask = self._initialize_mask()

    def __str__(self):
        s = "Finite model, built as '{0}' ".format(self._mode)
        if self._mode == 'tb':
            s += "from a {0} instance.\n".format("TBmodels" if isinstance(self._model, tbm.Model) else "PythTB")
            s += "The size of the sample is {0}x{1}, and has {2} atoms per unit cell.\n".format(self._nx_sites, self._ny_sites, self._atoms_uc)
            s += "The model {0} spinful, and it {1} disordered.".format("is" if self._spinful else "is not", "is" if self._disordered else "is not")
        else:
            s += "from external files.\n"
            s += "The model {0} spinful, and it {1} disordered.".format("is" if self._spinful else "is not", "is" if self._disordered else "is not")
        return s

    #################################################
    # Initialization of class variables
    #################################################

    def _get_positions(self):
        """
        Returns the cartesian coordinates of the states in a finite sample
        """
        if isinstance(self._model, tbm.Model):
            return _tbmodels.get_positions(self._model, self._nx_sites, self._ny_sites).T
        elif isinstance(self._model, ptb.tb_model):
            if not self._spinful:
                return _pythtb.orb_cart(self._model).T
            else:
                tmp = _pythtb.orb_cart(self._model).T
                newx = np.repeat(tmp[0], 2)
                newy = np.repeat(tmp[1], 2)
                return np.array([newx, newy])
        else:
            raise NotImplementedError("Invalid model instance.")

    def _load_positions(self, filename):
        """
        Read lattice positions from external files
        """
        return [np.loadtxt(filename[0], unpack = True), np.loadtxt(filename[1], unpack = True)]

    def _get_spin(self):
        """
        Returns the spin coordinates of the states in a finite sample
        """
        if isinstance(self._model, tbm.Model):
            return np.diag([1, -1, 1, -1] * int(self._model.size / self._atoms_uc))
        elif isinstance(self._model, ptb.tb_model):
            return np.diag([1, -1] * self._model.get_num_orbitals())
        else:
            raise NotImplementedError("Invalid model instance.")

    def _load_spin(self, filename):
        """
        Read spin matrix from external file
        """
        return np.loadtxt(filename, unpack = True)

    def _get_hamiltonian(self):
        """
        Returns the hamiltonian matrix
        """
        if isinstance(self._model, tbm.Model):
            return self._model.hamilton([0, 0], convention = 1), self._model.occ
        elif isinstance(self._model, ptb.tb_model):
            ham = self._model._gen_ham()
            occ = self._model.get_num_orbitals() if self._spinful else self._model.get_num_orbitals() // 2
            return ham if not self._spinful else ham.reshape((2 * occ, 2 * occ)), occ
        else:
            raise NotImplementedError("Invalid model instance.")

    def _load_hamiltonian(self, filename):
        """
        Read hamiltonian matrix from external file
        """
        ham = np.loadtxt(filename, unpack = True)
        return ham, int(len(ham) / 2)

    def _calc_uc_vol(self):
        """
        Returns the volume of the unit cell
        """
        if isinstance(self._model, tbm.Model):
            return _tbmodels._uc_vol(self._model)
        elif isinstance(self._model, ptb.tb_model):
            return _pythtb._uc_vol(self._model)
        else:
            raise NotImplementedError("Invalid model instance")

    def _calc_atoms_uc(self):
        """
        Returns the number of atoms per unit cell
        """
        if isinstance(self._model, tbm.Model):
            return int(self._model.size / (self._nx_sites * self._ny_sites))
        elif isinstance(self._model, ptb.tb_model):
            return int(self._model.get_num_orbitals() / (self._nx_sites * self._ny_sites)) * (2 if self._spinful else 1)
        else:
            raise NotImplementedError("Invalid model instance.")

    def _initialize_mask(self):
        """
        Initialize a mask for the disordered case, saying which indices to keep while evaluating the lattice
        """
        if isinstance(self._model, tbm.Model):
            return np.array([True for _ in range(self._model.size)])
        elif isinstance(self._model, ptb.tb_model):
            return np.array([True for _ in range(self._model.get_num_orbitals() * (2 if self._spinful else 1))])
        else:
            raise NotImplementedError("Invalid model instance.")

    #################################################
    # Protected variables getters
    #################################################

    @property
    def positions(self):
        return np.array([self._rx, self._ry])
    
    @property
    def atoms_uc(self):
        return self._atoms_uc
    
    @property
    def model(self):
        return self._model
    
    @property
    def lattice_dimension(self):
        return [self._nx_sites, self._ny_sites]

    #################################################
    # Lattice functions
    #################################################

    def add_onsite_disorder(self, w : float = 0, seed : int = None):
        """
        Add onsite (Anderson) disorder to the specified model. The disorder amplitude per site is taken randomly in [-w/2, w/2].

            Args:
            - w : disorder amplitude
            - seed : seed for random number generator

            Returns:
            - model : the disordered model
        """
        if w != 0: self._disordered = True
        if isinstance(self._model, tbm.Model):
            self._model = _tbmodels.onsite_disorder(self._model, w, spinstates = 2 if self._spinful else 1, seed = seed)
        elif isinstance(self._model, ptb.tb_model):
            self._model = _pythtb.onsite_disorder(self._model, w, spinstates = 2 if self._spinful else 1, seed = seed)
        else:
            raise NotImplementedError("Invalid model instance.")

    def add_vacancies(self, vacancies_list):
        """
        Add vacancies in the systems by removing a site in the lattice.

        Args:
            - vacancies_list: a list of [cell_x, cell_y, basis] pointing to the cell and atom to be removed. Multiple values at once are allowed
        """
        ham = self._hamiltonian
        pos = [self._rx, self._ry]
        sz = self._sz
        nocc = self._nocc

        # Convert vacancies_list to internal indexing
        if np.array(vacancies_list).shape == (3,):
            vacancies = _xy_to_index(self, vacancies_list[0], vacancies_list[1], vacancies_list[2])
        else:
            vacancies = []
            for hl in vacancies_list:
                vacancies.append(_xy_to_index(self, hl[0], hl[1], hl[2]))

        # Remove the selected sites
        if np.array(vacancies).shape == ():
            self._mask[vacancies] = False
            if not self._spinful:
                ham = np.delete(ham, holes, 0)
                ham = np.delete(ham, holes, 1)
                newx = np.delete(pos[0], holes)
                newy = np.delete(pos[1], holes)
                if sz is not None:
                    sz = np.delete(sz, holes, 0)
                    sz = np.delete(sz, holes, 1)
                nocc -= 1
            else:
                newx = pos[0]; newy = pos[1]
                self._mask[holes + 1] = False
                for _ in range(2):
                    ham = np.delete(ham, holes, 0)
                    ham = np.delete(ham, holes, 1)
                    newx = np.delete(newx, holes)
                    newy = np.delete(newy, holes)
                    if sz is not None:
                        sz = np.delete(sz, holes, 0)
                        sz = np.delete(sz, holes, 1)
                pos = np.array([newx, newy]).T
                nocc -= 2
        else:
            holes = np.sort(vacancies)[::-1]
            newx = pos[0]; newy = pos[1]
            for h in vacancies:
                self._mask[h] = False
                if not self._spinful:
                    ham = np.delete(ham, h, 0)
                    ham = np.delete(ham, h, 1)
                    newx = np.delete(newx, h)
                    newy = np.delete(newy, h)
                    if sz is not None:
                        sz = np.delete(sz, h, 0)
                        sz = np.delete(sz, h, 1)
                    nocc -= 1
                else:
                    self._mask[h] = False
                    self._mask[h + 1] = False
                    for _ in range(2):
                        ham = np.delete(ham, h, 0)
                        ham = np.delete(ham, h, 1)
                        newx = np.delete(newx, h)
                        newy = np.delete(newy, h)
                        if sz is not None:
                            sz = np.delete(sz, h, 0)
                            sz = np.delete(sz, h, 1)
                    pos = np.array([newx, newy]).T
                    nocc -= 2
        
        self._hamiltonian = ham
        self._rx = newx; self._ry = newy
        self._sz = sz
        self._nocc = nocc
        self._disordered = True

    #################################################
    # Local topological markers
    #################################################

    def local_chern_marker(self, direction : int = None, start : int = 0, return_projector : bool = False, projector : np.ndarray = None, macroscopic_average : bool = False, cutoff : float = 0.8):
        """
        Evaluate the Chern marker on the whole lattice if direction is None. If direction is not None evaluates the Z Chern marker along direction starting from start.
        
            Args:
            - direction : direction along which compute the local Chern marker, default is None (returns the whole lattice Chern marker), allowed values are 0 for 'x' direction and 1 for 'y' direction
            - start : if direction is not None, is the coordinate of the unit cell at the start of the evaluation of the Chern marker
            - return_projector : if True, returns the ground state projector at the end of the calculation, default is False
            - projector : input the ground state projector to be used in the calculation. Default is None, which means it is computed from the model
            - macroscopic_average : if True, returns the local Chern marker averaged over a radius equal to the cutoff
            - cutoff : cutoff set for the calculation of averages

            Returns:
            - lattice_chern : local Chern marker of the whole lattice if direction is None
            - lcm_direction : local Chern marker along direction starting from start
            - projector : ground state projector, returned if return_projector is set True (default is False)
        """

        # Check input variables
        if direction not in [None, 0, 1]:
            raise RuntimeError("Direction allowed are None, 0 (which stands for x), and 1 (which stands for y)")
        
        if direction is not None:
            if direction == 0:
                if start not in range(self._ny_sites): raise RuntimeError("Invalid start parameter (must be within [0, ny_sites - 1])")
            else:
                if start not in range(self._nx_sites): raise RuntimeError("Invalid start parameter (must be within [0, nx_sites - 1])")

        if projector is None:
            # Eigenvectors at \Gamma
            _, eigenvecs = la.eigh(self._hamiltonian)

            # Build the ground state projector
            gs_projector = contract("ji,ki->jk", eigenvecs[:, :self._nocc], eigenvecs[:, :self._nocc].conjugate())
        else:
            gs_projector = projector

        # Chern marker operator
        chern_operator = np.imag(gs_projector @ commutator(np.diag(self._rx), gs_projector) @ commutator(np.diag(self._ry), gs_projector))
        chern_operator *= -4 * np.pi / self._uc_vol

        # If macroscopic average I have to compute the lattice values with the averages first
        if macroscopic_average or self._disordered:
            lattice_chern = _average_over_radius(self, np.diag(chern_operator), cutoff)
        
        if direction is not None:
            # Evaluate index of the selected direction
            indices = _xy_to_line(self, 'x' if direction == 1 else 'y', start)

            # If macroscopic average consider the averaged lattice, else the Chern operator
            if macroscopic_average or self._disordered:
                lcm_direction = [lattice_chern[indices[i]] for i in range(len(indices))]
            else:
                lcm_direction = [np.sum([chern_operator[indices[self._atoms_uc * i + j], indices[self._atoms_uc * i + j]] for j in range(self._atoms_uc)]) for i in range(int(len(indices) / self._atoms_uc))]
            
            if not return_projector:
                return np.array(lcm_direction)
            else:
                return np.array(lcm_direction), gs_projector

        if not macroscopic_average and not self._disordered:
            lattice_chern = partialtrace(chern_operator, self._atoms_uc)
            lattice_chern = np.repeat(lattice_chern, self._atoms_uc)

        if not return_projector:
            return np.array(lattice_chern)
        else:
            return np.array(lattice_chern), gs_projector
        
    def local_spin_chern_marker(self, direction : int = None, start : int = 0, return_projector : bool = None, projector : np.ndarray = None, macroscopic_average : bool = False, cutoff : float = 0.8):
        """
        Evaluate the spin Chern marker on the whole lattice if direction is None. If direction is not None evaluates the spin Chern marker along direction starting from start.
            
            Args:
            - direction: direction along which compute the local spin Chern marker, default is None (returns the whole lattice spin Chern marker), allowed values are 0 for 'x' direction and 1 for 'y' direction
            - start: if direction is not None, is the coordinate of the unit cell at the start of the evaluation of the spin Chern marker
            - return_projector : if True, returns a list of the two individual "positive" P+ and "negative" P- projectors at the end of the calculation, default is False
            - projector : input the list of the individual "positive" P+ and "negative" P- projectors to be used in the calculation. Default is None, which means they are computed from the model
            - macroscopic_average : if True, returns the local spin Chern marker averaged over a radius equal to the cutoff
            - cutoff : cutoff set for the calculation of averages

            Returns:
            - lattice_chern: local spin Chern marker of the whole lattice if direction is None
            - lcm_direction: local spin Chern marker along direction starting from start
            - projector : list composed by the individual "positive" P+ and "negative" P- projectors, returned if return_projector is set True (default is False)
        """
        if not self._spinful:
            raise RuntimeError("Cannot evaluate the local spin Chern marker for a spinless model.")

        # Check input variables
        if direction not in [None, 0, 1]:
            raise RuntimeError("Direction allowed are None, 0 (which stands for x), and 1 (which stands for y)")
        
        if direction is not None:
            if direction == 0:
                if start not in range(self._ny_sites): raise RuntimeError("Invalid start parameter (must be within [0, ny_sites - 1])")
            else:
                if start not in range(self._nx_sites): raise RuntimeError("Invalid start parameter (must be within [0, nx_sites - 1])")

        if projector is None:
            # Evaluate the model to get the eigenvectors
            _, eigenvecs = la.eigh(self._hamiltonian)
            canonical_to_H = eigenvecs

            # S^z and PS^zP matrix in the eigenvector basis
            pszp = canonical_to_H.T[:self._nocc, :].conjugate() @ (self._sz @ canonical_to_H)[:, :self._nocc]
            _, evecs = la.eigh(pszp)
            
            # Eigevectors by row
            evecs = (canonical_to_H[:, :self._nocc] @ evecs).conjugate()

            # Now I build the projector onto the lower and higher eigenvalues
            lowerproj = contract("ki,ji->kj", evecs[:, :int(0.5 * evecs.shape[1])], evecs[:, :int(0.5 * evecs.shape[1])].conjugate())
            higherproj = contract("ki,ji->kj", evecs[:, int(0.5 * evecs.shape[1]):], evecs[:, int(0.5 * evecs.shape[1]):].conjugate())
        else:
            higherproj = projector[0]
            lowerproj = projector[1]

        # Position operator in tight-binding approximation (site orbitals basis)
        rx = np.diag(self._rx); ry = np.diag(self._ry)

        # Chern marker operator
        chern_operator_plus = np.imag(higherproj @ commutator(rx, higherproj) @ commutator(ry, higherproj))
        chern_operator_minus = np.imag(lowerproj @ commutator(rx, lowerproj) @ commutator(ry, lowerproj))
        chern_operator_plus *= -4 * np.pi / self._uc_vol
        chern_operator_minus *= -4 * np.pi / self._uc_vol

        # If macroscopic average I have to compute the lattice values with the averages first
        if macroscopic_average or self._disordered:
            chernmarker_plus = _average_over_radius(self, np.diag(chern_operator_plus), cutoff)
            chernmarker_minus = _average_over_radius(self, np.diag(chern_operator_minus), cutoff)
        
        if direction is not None:
            # Evaluate index of the selected direction
            indices = _xy_to_line(self, 'x' if direction == 1 else 'y', start)

            # If macroscopic average consider the averaged lattice, else the Chern operators
            if macroscopic_average or self._disordered:
                lcm_plus = [chernmarker_plus[indices[i]] for i in range(len(indices))]
                lcm_minus = [chernmarker_minus[indices[i]] for i in range(len(indices))]
            else:
                lcm_plus = [np.sum([chern_operator_plus[indices[self._atoms_uc * i + j], indices[self._atoms_uc * i + j]] for j in range(self._atoms_uc)]) for i in range(int(len(indices) / self._atoms_uc))]
                lcm_minus = [np.sum([chern_operator_minus[indices[self._atoms_uc * i] + j, indices[self._atoms_uc * i] + j] for j in range(self._atoms_uc)]) for i in range(int(len(indices) / self._atoms_uc))]
            
            lcm_direction = [np.fmod(0.5 * (lcm_plus[i] - lcm_minus[i]), 2) for i in range(len(lcm_plus))]

            if not return_projector:
                return np.array(np.abs(lcm_direction))
            else:
                return np.array(np.abs(lcm_direction)), [higherproj, lowerproj]
        
        # If not macroscopic averages I sum the values of the Chern operators of the unit cell
        if not macroscopic_average and not self._disordered:
            chernmarker_plus = [np.sum([chern_operator_plus[self._atoms_uc * i + j, self._atoms_uc * i + j] for j in range(self._atoms_uc)]) for i in range(int(len(chern_operator_plus) / self._atoms_uc))]
            chernmarker_minus = [np.sum([chern_operator_minus[self._atoms_uc * i + j, self._atoms_uc * i + j] for j in range(self._atoms_uc)]) for i in range(int(len(chern_operator_minus) / self._atoms_uc))]
            chernmarker_plus = np.repeat(chernmarker_plus, self._atoms_uc)
            chernmarker_minus = np.repeat(chernmarker_minus, self._atoms_uc)

        lattice_chern = np.fmod(0.5 * (np.array(chernmarker_plus) - np.array(chernmarker_minus)), 2)
        if not return_projector:
            return np.array(np.abs(lattice_chern))
        else:
            return np.array(np.abs(lattice_chern)), [higherproj, lowerproj]

    def localization_marker(self, direction : int = None, start : int = 0, return_projector : bool = None, projector : np.ndarray = None, macroscopic_average : bool = False, cutoff : float = 0.8):
        """
        Evaluate the localization marker on the whole lattice if direction is None. If direction is not None evaluates the localization marker along direction starting from start.
            
            Args:
            - direction : direction along which compute the local localization marker, default is None (returns the whole lattice localization marker), allowed values are 0 for 'x' direction and 1 for 'y' direction
            - start : if direction is not None, is the coordinate of the unit cell at the start of the evaluation of the localization marker
            - return_projector : if True, returns the ground state projector at the end of the calculation, default is False
            - projector : input the ground state projector to be used in the calculation. Default is None, which means it is computed from the model
            - macroscopic_average : if True, returns the local spin Chern marker averaged over a radius equal to the cutoff
            - cutoff : cutoff set for the calculation of averages

            Returns:
            - lattice_loc : local localization marker of the whole lattice if direction is None
            - loc_direction : local localization marker along direction starting from start if direction is not None
            - projector : ground state projector, returned if return_projector is set True (default is False)
        """

        # Check input variables
        if direction not in [None, 0, 1]:
            raise RuntimeError("Direction allowed are None, 0 (which stands for x), and 1 (which stands for y)")
        
        if direction is not None:
            if direction == 0:
                if start not in range(self._ny_sites): raise RuntimeError("Invalid start parameter (must be within [0, ny_sites - 1])")
            else:
                if start not in range(self._nx_sites): raise RuntimeError("Invalid start parameter (must be within [0, nx_sites - 1])")

        if projector is None:
            # Eigenvectors at \Gamma
            _, eigenvecs = la.eigh(self._hamiltonian)

            # Build the ground state projector
            gs_projector = contract("ji,ki->jk", eigenvecs[:, :int(0.5 * len(eigenvecs))], eigenvecs[:, :int(0.5 * len(eigenvecs))].conjugate())
        else:
            gs_projector = projector

        # Reduced coordinate
        if isinstance(self._model, tbm.Model):
            inv = la.inv(self._model.uc)
        elif isinstance(self._model, ptb.tb_model):
            inv = la.inv(self._model.get_lat())
        else:
            raise NotImplementedError("Invalid model instance.")
        positions = np.array([inv @ np.array([self._rx[i], self._ry[i]]) for i in range(len(self._rx))])

        # Position operator on a square lattice (reduced coordinate adjusted by the dimension of the sample)
        rx = np.diag(positions[:, 0]); ry = np.diag(positions[:, 1])

        # Local marker operator
        commxgsp = commutator(rx, gs_projector); commygsp = commutator(ry, gs_projector)
        localization_operator = -np.real(gs_projector @ commxgsp @ commxgsp) - np.real(gs_projector @ commygsp @ commygsp)

        # If macroscopic average I have to compute the lattice values with the averages first
        if macroscopic_average or self._disordered:
            lattice_loc = _average_over_radius(self, np.diag(localization_operator), cutoff)
        
        if direction is not None:
            # Evaluate index of the selected direction
            indices = _xy_to_line(self, 'x' if direction == 1 else 'y', start)

            # If macroscopic average consider the averaged lattice, else the localization operator
            if macroscopic_average or self._disordered:
                loc_direction = [lattice_loc[indices[i]] for i in range(len(indices))]
            else:
                loc_direction = [np.sum([localization_operator[indices[self._atoms_uc * i + j], indices[self._atoms_uc * i + j]] for j in range(self._atoms_uc)]) for i in range(int(len(indices) / self._atoms_uc))]
            
            if not return_projector:
                return np.array(loc_direction)
            else:
                return np.array(loc_direction), gs_projector

        if not macroscopic_average and not self._disordered:
            lattice_loc = partialtrace(localization_operator, self._atoms_uc)
            lattice_loc = np.repeat(lattice_loc, self._atoms_uc)

        if not return_projector:
            return np.array(lattice_loc)
        else:
            return np.array(lattice_loc), gs_projector

##################################################################################################
# Finite model builder
##################################################################################################

def make_finite(model, nx_sites : int, ny_sites : int):
    """
    Make a finite mdoel along x and y direction by first cutting on the y direction and then on the x. This convention has been used to track the positions in the functions

        Args:
        - model : instance of the model, which should be periodic in both x and y direction
        - nx_sites, ny_sites : number of sites of the finite sample in both directions

        Returns:
        - model : the finite model
    """

    if isinstance(model, tbm.Model):
        return _tbmodels.make_finite(model, nx_sites, ny_sites)
    elif isinstance(model, ptb.tb_model):
        return _pythtb.make_finite(model, nx_sites, ny_sites)
    else:
        raise NotImplementedError("Invalid model instance")

##################################################################################################
# Lattice sites indexing
##################################################################################################

def _xy_to_line(fmodel : FiniteModel, fixed_coordinate : str, xy : int):
    """
    Returns the indices of the sites of the lattice keeping fixed the direction fixed_coordinate starting from xy
    """
    if fixed_coordinate == 'x':
        return np.array([fmodel._atoms_uc * xy * fmodel._ny_sites + i for i in range(fmodel._atoms_uc * fmodel._ny_sites) if fmodel._mask[fmodel._atoms_uc * xy * fmodel._ny_sites + i]]).flatten().tolist()
    elif fixed_coordinate == 'y':
        indices = []
        for i in range(fmodel._nx_sites):
            for j in range(fmodel._atoms_uc):
                if fmodel._mask[fmodel._atoms_uc * xy + fmodel._atoms_uc * fmodel._ny_sites * i + j]:
                    indices.append(fmodel._atoms_uc * xy + fmodel._atoms_uc * fmodel._ny_sites * i + j)
        return np.array(indices)
    else:
        raise RuntimeError("Direction not allowed, only 'x' or 'y'")
    
def _xy_to_index(fmodel : FiniteModel, cellx : int, celly : int, basis : int):
    return fmodel._ny_sites * fmodel._atoms_uc * cellx + fmodel._atoms_uc * celly + basis * (2 if fmodel._spinful else 1)

##################################################################################################
# Disorder average handler
##################################################################################################

def _lattice_contraction(fmodel : FiniteModel, cutoff : float):
    """
    Defines which atomic sites must be contracted on one site, for each site of the lattice
    """
    contraction = []
    def within_range(current, trial):
        return True if (fmodel._rx[current] - fmodel._rx[trial]) ** 2 + (fmodel._ry[current] - fmodel._ry[trial]) ** 2 - cutoff ** 2 < 1e-6 else False

    for current in range(len(fmodel._rx)):
        contraction.append([trial for trial in range(len(fmodel._rx)) if within_range(current, trial)])

    return contraction

def _average_over_radius(fmodel : FiniteModel, vals, cutoff : float):
    """
    Average vals over the contraction of the lattice defined by the cutoff radius
    """
    return_vals = []
    contraction = _lattice_contraction(fmodel, cutoff)

    # Macroscopic average within a certain radius
    for current in range(len(fmodel._rx)):
        tmp = [vals[ind] for ind in contraction[current]]
        if not len(tmp) == 0:
            return_vals.append(np.sum(tmp) / (len(tmp) / fmodel._atoms_uc))
        else:
            raise RuntimeError("Unexpected error occourred in counting the neighbors of a lattice site, there are none")

    return np.array(return_vals)