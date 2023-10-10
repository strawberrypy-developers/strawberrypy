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

    def local_chern_marker(self, direction : int = None, start : int = 0, return_projector : bool = False, input_projector : np.ndarray = None, macroscopic_average : bool = False, cutoff : float = 0.8):
        """
        Evaluate the local Chern marker on the whole lattice if direction is None. If direction is not None evaluates the Chern marker along direction starting from start.
        
        Args:
            - direction : direction along which compute the local Chern marker, default is None (returns the marker on the whole lattice), allowed values are 0 for 'x' direction and 1 for 'y' direction
            - start : if direction is not None, is the coordinate of the unit cell at the start of the evaluation of the Chern marker
            - return_projector : if True, returns the ground state projector at the end of the calculation, default is False
            - input_projector : input the ground state projector to be used in the calculation. Default is None, which means it is computed from the model
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

        if input_projector is None:
            # Eigenvectors at \Gamma
            _, eigenvecs = la.eigh(self.hamiltonian)

            # Build the ground state projector
            gs_projector = contract("ji,ki->jk", eigenvecs[:, :self.n_occ], eigenvecs[:, :self.n_occ].conjugate())
        else:
            gs_projector = input_projector

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
        

    def local_spin_chern_marker(self, direction : int = None, start : int = 0, return_projector : bool = None, input_projector : np.ndarray = None, macroscopic_average : bool = False, cutoff : float = 0.8, check_gap : bool = False):
        """
        Evaluate the local spin Chern marker on the whole lattice if direction is None. If direction is not None evaluates the spin Chern marker along direction starting from start.
            
        Args:
            - direction: direction along which compute the local spin Chern marker, default is None (returns the marker on the whole lattice), allowed values are 0 for 'x' direction and 1 for 'y' direction
            - start: if direction is not None, is the coordinate of the unit cell at the start of the evaluation of the spin Chern marker
            - return_projector : if True, returns a list of the two individual "positive" P+ and "negative" P- projectors at the end of the calculation, default is False
            - input_projector : input the list of the individual "positive" P+ and "negative" P- projectors to be used in the calculation. Default is None, which means they are computed from the model
            - macroscopic_average : if True, returns the local spin Chern marker averaged over a radius equal to the cutoff
            - cutoff : cutoff set for the calculation of averages
            - check_gap : if True, checks that the gap of PSzP does not close (default is False)

        Returns:
            - lattice_chern : local spin Chern marker of the whole lattice if direction is None
            - lcm_direction : local spin Chern marker along direction starting from start
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

        if input_projector is None:
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
            higherproj = input_projector[0]
            lowerproj = input_projector[1]

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


    def localization_marker(self, direction : int = None, start : int = 0, return_projector : bool = None, input_projector : np.ndarray = None, macroscopic_average : bool = False, cutoff : float = 0.8):
        """
        Evaluate the localization marker on the whole lattice if direction is None. If direction is not None evaluates the localization marker along direction starting from start.
            
        Args:
            - direction : direction along which compute the local localization marker, default is None (returns the whole lattice localization marker), allowed values are 0 for 'x' direction and 1 for 'y' direction
            - start : if direction is not None, is the coordinate of the unit cell at the start of the evaluation of the localization marker
            - return_projector : if True, returns the ground state projector at the end of the calculation, default is False
            - input_projector : input the ground state projector to be used in the calculation. Default is None, which means it is computed from the model
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

        if input_projector is None:
            # Eigenvectors at \Gamma
            _, eigenvecs = la.eigh(self.hamiltonian)

            # Build the ground state projector
            gs_projector = contract("ji,ki->jk", eigenvecs[:, :int(0.5 * len(eigenvecs))], eigenvecs[:, :int(0.5 * len(eigenvecs))].conjugate())
        else:
            gs_projector = input_projector

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
        

    def local_Z2_marker(self, direction : int = None, start : int = 0, return_projector : bool = None, input_projector = None, macroscopic_average : bool = False, cutoff : float = 0.8, sd_minimization : bool = False, sd_beta : float = 0.001, sd_epsilon : float = 1e-6, sd_maxiter : int = 200):
        """
        Evaluate the local Z2 marker on the whole lattice if direction is None. If direction is not None evaluates the local Z2 marker along direction starting from start.
            
        Args:
            - direction: direction along which compute the local Z2 marker, default is None (returns the marker on the whole lattice), allowed values are 0 for 'x' direction and 1 for 'y' direction
            - start: if direction is not None, is the coordinate of the unit cell at the start of the evaluation of the Z2 marker
            - return_projector : if True, returns a list of the two individual P1 and P2 projectors at the end of the calculation, default is False
            - input_projector : input the list of the individual P1 and P2 projectors to be used in the calculation. Default is None, which means they are computed from the model
            - macroscopic_average : if True, returns the local Z2 marker averaged over a radius equal to the cutoff
            - cutoff : cutoff set for the calculation of averages
            - sd_minimization : if True, a steepest descent minimization is performed after the division into time-reversal-conjugated subspaces (default is False)
            - sd_beta : the "length" of a step of a single of steepest descent minimization
            - sd_epsilon : threshold for convergence of the steepest descent alrogithm
            - sd_maxiter : maximum number of iterations allowed for the steepest descent procedure

        Returns:
            - z2marker : local Z2 marker of the whole lattice if direction is None
            - lz2_direction : local Z2 marker along direction starting from start
            - projector : list composed by the individual P1 and P2 projectors, returned if return_projector is set True (default is False)
        """
        # Check input variables
        if direction not in [None, 0, 1]:
            raise RuntimeError("Direction allowed are None, 0 (which stands for x), and 1 (which stands for y)")
        
        if direction is not None:
            if direction == 0:
                if start not in range(self.Ly): raise RuntimeError("Invalid start parameter (must be within [0, ny_sites - 1])")
            else:
                if start not in range(self.Lx): raise RuntimeError("Invalid start parameter (must be within [0, nx_sites - 1])")

        if input_projector is None:
            # Evaluate the model to get the eigenvectors
            eigenvals, eigenvecs = la.eigh(self.hamiltonian)

            # VERSION 2
            eigenvecs_projected = self._delta_projection(eigenvecs[:, :self.n_occ])
            vectors, _ = self._time_reversal_separation(eigenvecs_projected, eigenvals[:self.n_occ])
            if sd_minimization: vectors = self._steepest_descent(vectors, sd_beta, sd_epsilon, sd_maxiter)

            # Sigma y matrix
            diagonal = np.array([[-1.j, 0] for jj in range(int(0.5 * len(eigenvecs)))], dtype = complex).flatten()
            sigma_y = np.diag(diagonal, 1) + np.diag(diagonal.conjugate(), -1)
            sigma_y = sigma_y[:len(eigenvecs), :len(eigenvecs)]

            # Compute the two projectors
            projector = contract("ik,ij->kj", vectors, vectors.conjugate())
            trprojector = contract("ik,ij->kj", np.array([1.j * sigma_y @ vi.conjugate() for vi in vectors]), np.array([1.j * sigma_y @ vi.conjugate() for vi in vectors]).conjugate())
        else:
            projector = input_projector[0]
            trprojector = input_projector[1]

        # Chern marker operator
        rxpc = self.r[0] @ projector - projector @ self.r[0]
        rypc = self.r[1] @ projector - projector @ self.r[1]
        lz2_operator_1 = np.imag(projector @ rxpc @ rypc)
        rxtpc = self.r[0] @ trprojector - trprojector @ self.r[0]
        rytpc = self.r[1] @ trprojector - trprojector @ self.r[1]
        lz2_operator_2 = np.imag(trprojector @ rxtpc @ rytpc)
        lz2_operator_1 *= -4 * np.pi / self.uc_vol
        lz2_operator_2 *= -4 * np.pi / self.uc_vol

        # If macroscopic average I have to compute the lattice values with the averages first
        if macroscopic_average:
            z2marker_1 = self._average_over_radius(np.diag(lz2_operator_1), cutoff)
            z2marker_2 = self._average_over_radius(np.diag(lz2_operator_2), cutoff)
        
        if direction is not None:
            # Evaluate index of the selected direction
            indices = self._xy_to_index('x' if direction == 1 else 'y', start)

            # If macroscopic average consider the averaged lattice, else the Chern operators
            if macroscopic_average:
                lz2_1 = [z2marker_1[int(indices[self.states_uc * i] / self.states_uc)] for i in range(int(len(indices) / self.states_uc))]
                lz2_2 = [z2marker_2[int(indices[self.states_uc * i] / self.states_uc)] for i in range(int(len(indices) / self.states_uc))]
            else:
                lz2_1 = [np.sum([lz2_operator_1[indices[self.states_uc * i + j], indices[self.states_uc * i + j]] for j in range(self.states_uc)]) for i in range(int(len(indices) / self.states_uc))]
                lz2_2 = [np.sum([lz2_operator_2[indices[self.states_uc * i + j], indices[self.states_uc * i + j]] for j in range(self.states_uc)]) for i in range(int(len(indices) / self.states_uc))]
            
            lz2_direction = [np.fmod(0.5 * (lz2_1[i] - lz2_2[i]), 2) for i in range(len(lz2_1))]

            if not return_projector:
                return np.array(np.abs(lz2_direction))
            else:
                return np.array(np.abs(lz2_direction)), [projector, trprojector]
        
        # If not macroscopic averages I sum the values of the Chern operators of the unit cell
        if not macroscopic_average:
            z2invariant_2 = [np.sum([lz2_operator_2[self.states_uc * i + j, self.states_uc * i + j] for j in range(self.states_uc)]) for i in range(int(len(lz2_operator_2) / self.states_uc))]
            z2invariant_1 = [np.sum([lz2_operator_1[self.states_uc * i + j, self.states_uc * i + j] for j in range(self.states_uc)]) for i in range(int(len(lz2_operator_1) / self.states_uc))]
            z2invariant_1 = np.repeat(z2invariant_1, self.states_uc)
            z2invariant_2 = np.repeat(z2invariant_2, self.states_uc)

        z2marker = np.fmod(0.5 * (np.array(z2invariant_2) - np.array(z2invariant_1)), 2)
        if not return_projector:
            return np.array(np.abs(z2marker))
        else:
            return np.array(np.abs(z2marker)), [projector, trprojector]
    
    #################################################
    # Local Z2 marker auxiliary functions
    #################################################

    def _time_reversal_separation(self, eigenvectors, eigenvalues):
        """
        Separates the eigenvectors into two time-reversal conjugated subspaces
        """
        
        def in_subspace(v, subspace):
            # Check whether the vector v is in the subspace (apart from a phase)
            for w in range(len(subspace)):
                if np.abs(np.abs(np.vdot(v, subspace[w])) - 1) < 1e-10: return True, w
            return False

        def degenerate_subspaces_dimension(evals):
            # Determine the dimensions of the degenerate subspaces
            dimensions = []
            dim = 1

            for i in range(1, len(evals)):
                if evals[i] - evals[i - 1] < 1e-10:
                    dim += 1
                else:
                    dimensions.append(dim)
                    dim = 1
            
            dimensions.append(dim)
            return np.array(dimensions, dtype = int)

        def orthogonalize(v, sub1, sub2):
            # Orthogonalize vectors belonging to the same subspace
            vec = np.copy(v)
            already_orthogonal = True
            
            # Check orthogonality in subspace 1
            for w in sub1:
                if np.abs(np.abs(np.vdot(vec, w)) - 1) > 1e-10: already_orthogonal = False

            # Check orthogonality in subspace 2
            for w in sub2:
                if np.abs(np.abs(np.vdot(vec, w)) - 1) > 1e-10: already_orthogonal = False

            if not already_orthogonal:
                # If the vector is not already orthogonal I orthogonalize wrt the two subspaces
                for i in range(len(sub1)):
                    alpha = np.vdot(sub1[i], vec)
                    vec -= alpha * sub1[i]

                    beta = np.vdot(sub2[i], vec)
                    vec -= beta * sub2[i]

                    if np.linalg.norm(vec) < 1e-10:
                        return np.zeros_like(vec)

                    vec /= np.linalg.norm(vec)
            return vec

        def is_generated_by(vec, sub1, sub2):
            # Check whether a vector can be written as a liear combinations of the vectors in other subspaces
            alphas = []; betas = []
            if len(sub1) > 0:
                for i in range(len(sub1)):
                    alphas.append(np.vdot(sub1[i], vec))
                    betas.append(np.vdot(sub2[i], vec))
                if np.allclose(alphas, np.zeros_like(alphas)) and np.allclose(betas, np.zeros_like(betas)): return False
                return True
            else:
                return False

        def check_subspace_orthogonality(v, w):
            # Check that the subspaces are orthogonal
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    if np.abs(np.vdot(v[i], v[j])) < 1e-10: continue
                    if np.abs(np.vdot(v[i], v[j])) > 1e-10:
                        raise RuntimeError("Vectors in the subspace are not orthogonal")
            for i in range(len(w)):
                for j in range(i + 1, len(w)):
                    if np.abs(np.vdot(w[i], w[j])) < 1e-10: continue
                    if np.abs(np.vdot(w[i], w[j])) > 1e-10:
                        raise RuntimeError("Vectors in the time reversal conjugated subspace are not orthogonal")
            for i in range(len(v)):
                for j in range(i + 1, len(w)):
                    if np.abs(np.vdot(v[i], w[j])) < 1e-10: continue
                    if np.abs(np.vdot(v[i], w[j])) > 1e-10:
                        raise RuntimeError("Subspaces are not orthogonal")
            return True

        # Eigenvectors by row
        revecs = np.copy(eigenvectors)

        # Split the eigenvectors in their degenerate subspaces
        subspaces_occ = degenerate_subspaces_dimension(eigenvalues)

        # Vectors of the half space to store (in rows)
        vectors = []; tr_vectors = []

        # Sigma y matrix
        diagonal = np.array([[-1.j, 0] for jj in range(revecs.shape[1])], dtype = complex).flatten()
        sigma_y = np.diag(diagonal, 1) + np.diag(diagonal.conjugate(), -1)
        sigma_y = sigma_y[:revecs.shape[1], :revecs.shape[1]]

        # Current state index
        newk = 0

        # Cycle over the number of degenerate subspaces
        for i in range(len(subspaces_occ)):
            if subspaces_occ[i] % 2 != 0: raise RuntimeError("Hamiltonian is not time reversal symmetryc since a degenerate subspace dimension is odd")

            k = newk

            # Eigenvector index
            newk += int(subspaces_occ[i])

            # If the dimension of the subspace is 2 I can choose one eigenvector and discard the other (perpendicular via TR symmetry)
            trvec = 1.j * sigma_y @ (revecs[k].conjugate())
            vectors.append(revecs[k])
            tr_vectors.append(trvec)
            if int(subspaces_occ[i]) == 2: continue

            # Otherwise I orthogonalize --- This should not happen
            subspace = []; tr_subspace = []
            subspace.append(vectors[-1]); tr_subspace.append(tr_vectors[-1])

            for j in range(subspaces_occ[i] - 1):
                vec = revecs[k + j + 1]

                # Check whether vec is already stored in subspace or time reversal subspace
                present = in_subspace(vec, subspace)
                present_tr = in_subspace(vec, tr_subspace)

                # If the vector is already stored continue
                if present or present_tr: continue

                # If it is not stored, I first check that it is orthogonal with the previous vectors
                vec = orthogonalize(vec, subspace, tr_subspace)
                if np.allclose(vec, np.zeros_like(vec)): continue
                if is_generated_by(vec, subspace, tr_subspace): continue

                # I compute the time reversal and store vec and trvec
                trvec = 1.j * sigma_y @ (vec.conjugate())

                subspace.append(vec)
                tr_subspace.append(trvec)
                vectors.append(vec)
                tr_vectors.append(trvec)

            if not check_subspace_orthogonality(subspace, tr_subspace):
                raise RuntimeError("Something went wrong: the subspace and its time reversal conjugated are not orthogonal")

        return np.array(vectors, dtype = complex), np.array(tr_vectors, dtype = complex)


    def _steepest_descent(self, vecs_orig, beta, epsilon, maxiterations):
        """
        Steepest descent minimization of the quadratic spread of the vectors
        """
        vecs = np.copy(vecs_orig)

        def compute_var(states):
            var = 0
            for psi in states:
                var += (np.vdot(psi, (self.r[0] ** 2) @ psi) - np.abs(np.vdot(psi, self.r[0] @ psi)) ** 2)
                var += (np.vdot(psi, (self.r[1] ** 2) @ psi) - np.abs(np.vdot(psi, self.r[1] @ psi)) ** 2)
            return var

        def compute_gradient(states):
            states = np.array(states)
            xprime = states.conjugate() @ (x @ states.T)
            yprime = states.conjugate() @ (y @ states.T)
            xd = np.diag(np.diag(xprime))
            yd = np.diag(np.diag(yprime))
            comx = (xprime - xd) @ xd - xd @ (xprime - xd)
            comy = (yprime - yd) @ yd - yd @ (yprime - yd)
            return 2 * (comx + comy)

        # Steepest descent minimizazion localizations
        diff = 1e+3; var = compute_var(vecs); iterations = 0
        minvar = var; saved = np.copy(vecs)
        while True:
            gradient = np.array(compute_gradient(vecs), dtype = np.complex128)

            g = la.expm(-beta * gradient)
            test = contract("ij,jk->ik", g, vecs)

            newvar = compute_var(test)
            diff = var - newvar
            var = newvar

            if var < minvar:
                minvar = var
                saved = np.copy(test)

            # Max iterations check
            if iterations > maxiterations: break
            if diff <= epsilon: break
            iterations += 1
            vecs = np.copy(test)

        return np.array(saved)


    def _delta_projection(self, vecs):
        """
        Projection onto trial orbitals in order to build MLWF
        """
        one = np.zeros(shape = (vecs.shape[0], vecs.shape[0]))
        projections = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, -1, 0]
        ])
        one = np.kron(np.eye(int(0.25*vecs.shape[0])), projections) / np.sqrt(2)
        amatrix = np.array([[np.vdot(vecs[:, m], one[:, 2 * n]) for n in range(vecs.shape[1])] for m in range(vecs.shape[1])], dtype = np.complex128)

        smatrix = np.array(la.sqrtm(la.pinv(amatrix.T.conjugate() @ amatrix)), dtype = np.complex128)
        return contract("ki,ij->kj", vecs, amatrix @ smatrix).T