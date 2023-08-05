import numpy as np
import tbmodels as tbm
import pythtb as ptb
import scipy.linalg as la
from opt_einsum import contract

from . import _tbmodels
from . import _pythtb
from .classes import Model

class FiniteModel(Model):
    """
    A class describing a finite model with either TBmodels or PythTB instances. Ability to add disorder and vacancies to the models,
    calculate local topological markers and localization markers.
    FiniteModel expects to have as input a PBC instance of TBmodels or PythTB, the number of cells along x and y direction and a 
    bool (spinful) which specifies if the model has spinful electrons;
    """

    def __init__(self, tbmodel = None, Lx : int = 1, Ly : int = 1, spinful : bool = False):
        # Store local variables
        self.Lx = Lx
        self.Ly = Ly
        self.model = self._make_finite(tbmodel)
        self.uc_vol = self._calc_uc_vol()

        # make_finite in PythTB delete informations about the dimensionaloty of the system
        self.dim = super()._get_dim(tbmodel)

        super().__init__(tbmodel = self.model,
                         spinful = spinful,
                         states_uc = super()._calc_states_uc(tbmodel, spinful),
                         dim = self.dim,
                         Lx = self.Lx,
                         Ly = self.Ly
                        )
        
        # The positions for TBModels must be rescaled by the supercell dimension
        self.cart_positions = self._get_positions()
        self.r = []
        for d in range(self.dim):
            self.r.append( np.diag(self.cart_positions[:, d]) )

    #################################################
    # Load functions
    #################################################

    def _get_positions(self):
        """
        Returns the cartesian coordinates of the states in a finite sample
        """
        if isinstance(self.model, tbm.Model):
            return _tbmodels.get_positions(self.model, self.Lx, self.Ly)
        elif isinstance(self.model, ptb.tb_model):
            return _pythtb.get_positions(self.model, self.spinful)
        else:
            raise NotImplementedError("Invalid model instance.")

    def _calc_uc_vol(self):
        """
        Returns the volume of a 2D unit cell
        """
        if isinstance(self.model, tbm.Model):
            return _tbmodels.calc_uc_vol(self.model)
        elif isinstance(self.model, ptb.tb_model):
            return _pythtb.calc_uc_vol(self.model)
        else:
            return NotImplementedError("Invalid model instance.")
        
        
    def _make_finite(self, model):
        """
        Returns an instance of a OBC model
        """
        if isinstance(model, tbm.Model):
            return _tbmodels.make_finite(model, self.Lx, self.Ly)
        elif isinstance(model, ptb.tb_model):
            return _pythtb.make_finite(model, self.Lx, self.Ly)
        else:
            return NotImplementedError("Invalid model instance.")
        
    #################################################
    # Local markers
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
                if start not in range(self.Ly): raise RuntimeError("Invalid start parameter (must be within [0, Ly - 1])")
            else:
                if start not in range(self.Lx): raise RuntimeError("Invalid start parameter (must be within [0, Lx - 1])")

        if len(self.r) != 2:
            raise NotImplementedError("The local Chern marker is not yet implemented for dimensionality different than 2")

        if projector is None:
            # Eigenvectors at \Gamma
            _, eigenvecs = la.eigh(self.hamiltonian)

            # Build the ground state projector
            gs_projector = contract("ji,ki->jk", eigenvecs[:, :self.n_occ], eigenvecs[:, :self.n_occ].conjugate())
        else:
            gs_projector = projector

        # Chern marker operator
        commut_rx_gsp = self.r[0] @ gs_projector - gs_projector @ self.r[0]
        commut_ry_gsp = self.r[1] @ gs_projector - gs_projector @ self.r[1]
        chern_operator = np.imag(gs_projector @ commut_rx_gsp @ commut_ry_gsp)
        chern_operator *= -4 * np.pi / self.uc_vol

        # If macroscopic average I have to compute the lattice values with the averages first
        if macroscopic_average or self.disordered:
            lattice_chern = self._average_over_radius(np.diag(chern_operator), cutoff)
        
        if direction is not None:
            # Evaluate index of the selected direction
            indices = self._xy_to_line('x' if direction == 1 else 'y', start)

            # If macroscopic average consider the averaged lattice, else the Chern operator
            if macroscopic_average or self.disordered:
                lcm_direction = [lattice_chern[indices[i]] for i in range(len(indices))]
            else:
                lcm_direction = [np.sum([chern_operator[indices[self.states_uc * i + j], indices[self.states_uc * i + j]] for j in range(self.states_uc)]) for i in range(int(len(indices) / self.states_uc))]
            
            if not return_projector:
                return np.array(lcm_direction)
            else:
                return np.array(lcm_direction), gs_projector

        if not macroscopic_average and not self.disordered:
            lattice_chern = [np.sum([chern_operator[i * self.states_uc + k, i * self.states_uc + k] for k in range(self.states_uc)]) for i in range(int(len(chern_operator) / self.states_uc))]
            lattice_chern = np.repeat(lattice_chern, self.states_uc)

        if not return_projector:
            return np.array(lattice_chern)
        else:
            return np.array(lattice_chern), gs_projector
        
    def local_spin_chern_marker(self, direction : int = None, start : int = 0, return_projector : bool = None, projector : np.ndarray = None, macroscopic_average : bool = False, cutoff : float = 0.8, check_gap : bool = False):
        """
        Evaluate the spin Chern marker on the whole lattice if direction is None. If direction is not None evaluates the spin Chern marker along direction starting from start.
            
        Args:
            - direction: direction along which compute the local spin Chern marker, default is None (returns the whole lattice spin Chern marker), allowed values are 0 for 'x' direction and 1 for 'y' direction
            - start: if direction is not None, is the coordinate of the unit cell at the start of the evaluation of the spin Chern marker
            - return_projector : if True, returns a list of the two individual "positive" P+ and "negative" P- projectors at the end of the calculation, default is False
            - projector : input the list of the individual "positive" P+ and "negative" P- projectors to be used in the calculation. Default is None, which means they are computed from the model
            - macroscopic_average : if True, returns the local spin Chern marker averaged over a radius equal to the cutoff
            - cutoff : cutoff set for the calculation of averages
            - check_gap : if True, checks that the gap of PSzP does not close (default is False)

        Returns:
            - lattice_chern: local spin Chern marker of the whole lattice if direction is None
            - lcm_direction: local spin Chern marker along direction starting from start
            - projector : list composed by the individual "positive" P+ and "negative" P- projectors, returned if return_projector is set True (default is False)
        """
        if not self.spinful:
            raise RuntimeError("Cannot evaluate the local spin Chern marker for a spinless model.")

        # Check input variables
        if direction not in [None, 0, 1]:
            raise RuntimeError("Direction allowed are None, 0 (which stands for x), and 1 (which stands for y)")
        
        if direction is not None:
            if direction == 0:
                if start not in range(self.Ly): raise RuntimeError("Invalid start parameter (must be within [0, ny_sites - 1])")
            else:
                if start not in range(self.Lx): raise RuntimeError("Invalid start parameter (must be within [0, nx_sites - 1])")

        if projector is None:
            # Evaluate the model to get the eigenvectors
            _, eigenvecs = la.eigh(self.hamiltonian)
            canonical_to_H = eigenvecs

            # S^z and PS^zP matrix in the eigenvector basis
            pszp = canonical_to_H.T[:self.n_occ, :].conjugate() @ (self.sz @ canonical_to_H)[:, :self.n_occ]
            evals, evecs = la.eigh(pszp)

            # Check that the gap in PSzP does not close
            if check_gap and evals[int(self.n_occ // 2)] - evals[int(self.n_occ // 2) - 1] < 1e-14:
                raise RuntimeError("PSzP gap closes!")
            
            # Eigevectors by row
            evecs = (canonical_to_H[:, :self.n_occ] @ evecs).conjugate()

            # Now I build the projector onto the lower and higher eigenvalues
            lowerproj = contract("ki,ji->kj", evecs[:, :int(0.5 * evecs.shape[1])], evecs[:, :int(0.5 * evecs.shape[1])].conjugate())
            higherproj = contract("ki,ji->kj", evecs[:, int(0.5 * evecs.shape[1]):], evecs[:, int(0.5 * evecs.shape[1]):].conjugate())
        else:
            higherproj = projector[0]
            lowerproj = projector[1]

        # Chern marker operator
        comm_rx_high = self.r[0] @ higherproj - higherproj @ self.r[0]
        comm_ry_high = self.r[1] @ higherproj - higherproj @ self.r[1]
        comm_rx_low = self.r[0] @ lowerproj - lowerproj @ self.r[0]
        comm_ry_low = self.r[1] @ lowerproj - lowerproj @ self.r[1]
        chern_operator_plus = np.imag(higherproj @ comm_rx_high @ comm_ry_high)
        chern_operator_minus = np.imag(lowerproj @ comm_rx_low @ comm_ry_low)
        chern_operator_plus *= -4 * np.pi / self.uc_vol
        chern_operator_minus *= -4 * np.pi / self.uc_vol

        # If macroscopic average I have to compute the lattice values with the averages first
        if macroscopic_average or self.disordered:
            chernmarker_plus = self._average_over_radius(np.diag(chern_operator_plus), cutoff)
            chernmarker_minus = self._average_over_radius(np.diag(chern_operator_minus), cutoff)
        
        if direction is not None:
            # Evaluate index of the selected direction
            indices = self._xy_to_line('x' if direction == 1 else 'y', start)

            # If macroscopic average consider the averaged lattice, else the Chern operators
            if macroscopic_average or self.disordered:
                lcm_plus = [chernmarker_plus[indices[i]] for i in range(len(indices))]
                lcm_minus = [chernmarker_minus[indices[i]] for i in range(len(indices))]
            else:
                lcm_plus = [np.sum([chern_operator_plus[indices[self.states_uc * i + j], indices[self.states_uc * i + j]] for j in range(self.states_uc)]) for i in range(int(len(indices) / self.states_uc))]
                lcm_minus = [np.sum([chern_operator_minus[indices[self.states_uc * i + j], indices[self.states_uc * i + j]] for j in range(self.states_uc)]) for i in range(int(len(indices) / self.states_uc))]
            
            lcm_direction = [np.fmod(0.5 * (lcm_plus[i] - lcm_minus[i]), 2) for i in range(len(lcm_plus))]

            if not return_projector:
                return np.array(np.abs(lcm_direction))
            else:
                return np.array(np.abs(lcm_direction)), [higherproj, lowerproj]
        
        # If not macroscopic averages I sum the values of the Chern operators of the unit cell
        if not macroscopic_average and not self.disordered:
            chernmarker_plus = [np.sum([chern_operator_plus[self.states_uc * i + j, self.states_uc * i + j] for j in range(self.states_uc)]) for i in range(int(len(chern_operator_plus) / self.states_uc))]
            chernmarker_minus = [np.sum([chern_operator_minus[self.states_uc * i + j, self.states_uc * i + j] for j in range(self.states_uc)]) for i in range(int(len(chern_operator_minus) / self.states_uc))]
            chernmarker_plus = np.repeat(chernmarker_plus, self.states_uc)
            chernmarker_minus = np.repeat(chernmarker_minus, self.states_uc)

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

        # BEWARE: THIS FUNCTION WORKS ONLY WITH TBMODELS AND PYTHTB UP TO NOW
        if self.model == None:
            raise NotImplementedError("The localization marker is implemented only for TBmodels and PythTB up to now.")

        # Check input variables
        if direction not in [None, 0, 1]:
            raise RuntimeError("Direction allowed are None, 0 (which stands for x), and 1 (which stands for y)")
        
        if direction is not None:
            if direction == 0:
                if start not in range(self.Ly): raise RuntimeError("Invalid start parameter (must be within [0, ny_sites - 1])")
            else:
                if start not in range(self.Lx): raise RuntimeError("Invalid start parameter (must be within [0, nx_sites - 1])")

        if projector is None:
            # Eigenvectors at \Gamma
            _, eigenvecs = la.eigh(self.hamiltonian)

            # Build the ground state projector
            gs_projector = contract("ji,ki->jk", eigenvecs[:, :int(0.5 * len(eigenvecs))], eigenvecs[:, :int(0.5 * len(eigenvecs))].conjugate())
        else:
            gs_projector = projector

        # Reduced coordinate
        if isinstance(self.model, tbm.Model):
            inv = la.inv(self.model.uc)
        elif isinstance(self.model, ptb.tb_model):
            inv = la.inv(self.model.get_lat())
        else:
            raise NotImplementedError("Invalid model instance.")
        positions = np.array([np.dot([self.r[0][i, i], self.r[1][i, i]], inv) for i in range(len(self.r[0]))])

        # Position operator on a square lattice (reduced coordinate adjusted by the dimension of the sample)
        rx = np.diag(positions[:, 0]); ry = np.diag(positions[:, 1])

        # Local marker operator
        commxgsp = rx @ gs_projector - gs_projector @ rx
        commygsp = ry @ gs_projector - gs_projector @ ry
        localization_operator = -np.real(gs_projector @ commxgsp @ commxgsp) - np.real(gs_projector @ commygsp @ commygsp)

        # If macroscopic average I have to compute the lattice values with the averages first
        if macroscopic_average or self.disordered:
            lattice_loc = self._average_over_radius(np.diag(localization_operator), cutoff)
        
        if direction is not None:
            # Evaluate index of the selected direction
            indices = self._xy_to_line('x' if direction == 1 else 'y', start)

            # If macroscopic average consider the averaged lattice, else the localization operator
            if macroscopic_average or self.disordered:
                loc_direction = [lattice_loc[indices[i]] for i in range(len(indices))]
            else:
                loc_direction = [np.sum([localization_operator[indices[self.states_uc * i + j], indices[self.states_uc * i + j]] for j in range(self.states_uc)]) for i in range(int(len(indices) / self.states_uc))]
            
            if not return_projector:
                return np.array(loc_direction)
            else:
                return np.array(loc_direction), gs_projector

        if not macroscopic_average and not self.disordered:
            lattice_loc = [np.sum([localization_operator[i * self.states_uc + k, i * self.states_uc + k] for k in range(self.states_uc)]) for i in range(int(len(localization_operator) / self.states_uc))]
            lattice_loc = np.repeat(lattice_loc, self.states_uc)

        if not return_projector:
            return np.array(lattice_loc)
        else:
            return np.array(lattice_loc), gs_projector
        
    #################################################
    # Utility functions
    #################################################

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

    def _average_over_radius(self, vals, cutoff : float):
        """
        Average vals over the contraction of the lattice defined by the cutoff radius
        """
        return_vals = []
        contraction = self._lattice_contraction(cutoff)

        # Macroscopic average within a certain radius
        for current in range(len(self.cart_positions[:, 0])):
            tmp = [vals[ind] for ind in contraction[current]]
            if not len(tmp) == 0:
                return_vals.append(np.sum(tmp) / (len(tmp) / self.states_uc))
            else:
                raise RuntimeError("Unexpected error occourred in counting the neighbors of a lattice site, there are none")

        return np.array(return_vals)