import numpy as np
import tbmodels as tbm
import pythtb as ptb
import scipy.linalg as la
from opt_einsum import contract

from . import _tbmodels
from . import _pythtb
from .common_func import *
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
        chern_operator = np.imag(gs_projector @ commutator(self.r[0], gs_projector) @ commutator(self.r[1], gs_projector))
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
            lattice_chern = partialtrace(chern_operator, self.states_uc)
            lattice_chern = np.repeat(lattice_chern, self.states_uc)

        if not return_projector:
            return np.array(lattice_chern)
        else:
            return np.array(lattice_chern), gs_projector
        
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