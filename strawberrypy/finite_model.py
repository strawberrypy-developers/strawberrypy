import numpy as np
import tbmodels as tbm
import pythtb as ptb
import scipy.linalg as la
from opt_einsum import contract

from . import _tbmodels
from . import _pythtb
from .classes import Model
from .utils import *

class FiniteModel(Model):
    r"""
    A class describing a finite (OBC) model built from either TBmodels or PythTB instances. It contains methods to add disorder and vacancies to the finite model, calculate local topological markers and the localization marker.
    
    Parameters
    ----------
        tbmodel : 
            Tight-binding model constructed from TBmodels or PythTB.
        Lx :
            Number of unit cells repeated along the :math:`\mathbf{a}_1` direction in the finite sample.
        Ly :
            Number of unit cells repeated along the :math:`\mathbf{a}_2` direction in the finite sample.
        spinful :
            Whether the model should be interpreted as spinful or not. Default is :python:`False`.
    """

    def __init__(self, tbmodel, Lx : int = 1, Ly : int = 1, spinful : bool = False):
        # Store local variables
        self.Lx = Lx
        self.Ly = Ly
        self.model = self._make_finite(tbmodel)
        self.uc_vol = self._calc_uc_vol()

        super().__init__(tbmodel = self.model,
                         spinful = spinful,
                         states_uc = super()._calc_states_uc(tbmodel, spinful),
                         Lx = self.Lx,
                         Ly = self.Ly
                        )
        
        # The positions for TBModels must be rescaled by the supercell dimension
        self.cart_positions = self._get_positions()
        self.r = []
        for d in range(self.dim_r):
            self.r.append( np.diag(self.cart_positions[:, d]) )

    #################################################
    # Load functions
    #################################################

    def _get_positions(self):
        r"""
        Returns the cartesian coordinates of the states in a finite sample.
        """
        if isinstance(self.model, tbm.Model):
            return _tbmodels.get_positions(self.model, self.Lx, self.Ly)
        elif isinstance(self.model, ptb.tb_model):
            return _pythtb.get_positions(self.model, self.spinful)
        else:
            raise NotImplementedError("Invalid model instance.")


    def _calc_uc_vol(self):
        r"""
        Returns the volume of a 2D unit cell.
        """
        if isinstance(self.model, tbm.Model):
            return _tbmodels.calc_uc_vol(self.model)
        elif isinstance(self.model, ptb.tb_model):
            return _pythtb.calc_uc_vol(self.model)
        else:
            raise NotImplementedError("Invalid model instance.")
        
        
    def _make_finite(self, model):
        r"""
        Returns a new instance of a model with open boundary conditions (removing periodic hoppings in the Hamiltonian).

        Parameters
        ----------
            model :
                Model instance.
        """
        if isinstance(model, tbm.Model):
            return _tbmodels.make_finite(model, self.Lx, self.Ly)
        elif isinstance(model, ptb.tb_model):
            return _pythtb.make_finite(model, self.Lx, self.Ly)
        else:
            raise NotImplementedError("Invalid model instance.")
        
    #################################################
    # Local markers
    #################################################

    def local_chern_marker(self, direction : int = None, start : int = 0, return_projector : bool = False, input_projector : np.ndarray = None, macroscopic_average : bool = False, cutoff : float = 0.8, smearing_temperature : float = 0.0, fermidirac_cutoff : float = 0.1):
        r"""
        Evaluate the local Chern marker provided in Ref. `Bianco-Resta (2011) <https://doi.org/10.1103/PhysRevB.84.241106>`_ on the whole lattice if :python:`direction == None`. If `:python:direction` is not :python:`None`, evaluate the local Chern marker along :python:`direction` starting from :python:`start`. Allowed directions are ``0`` (meaning along :math:`\mathbf{a}_1`), and ``1`` (meaning along :math:`\mathbf{a}_2`).
        
        Parameters
        ----------
            direction :
                Direction along which the local Chern marker is computed. Default is :python:`None` (returns the marker on the whole lattice). Allowed directions are ``0`` (meaning along :math:`\mathbf{a}_1`), and ``1`` (meaning along :math:`\mathbf{a}_2`).
            start :
                If :python:`direction` is not :python:`None`, indicates the coordinate of the unit cell from which the evaluation of the local Chern marker starts. For instance, if interested on the value of the local marker along the :math:`\mathbf{a}_1` direction at half height, it should be set :python:`direction = 0` and :python:`start = Ly // 2`.
            return_projector :
                If :python:`True`, returns the ground state projector. Default is :python:`False`.
            input_projector :
                Possibility to provide the ground state projector to be used in the calculation as input. Default is :python:`None`, which means that the ground state projector is computed from the model stored in the class.
            macroscopic_average :
                If :python:`True`, returns the local Chern marker averaged in real space over a radius equal to :python:`cutoff`. Default is :python:`False`.
            cutoff :
                Cutoff set for the calculation of the macroscopic average in real space of the local Chern marker.
            smearing_temperature :
                Set a fictitious temperature :math:`T_s` to be used when weighting the eigenstates of the Hamiltonian comprising the ground state projector. In particular, the ground state projector is computed as :math:`\mathcal P=\sum_{n}f(\epsilon_n, T_s, \mu)|u_n\rangle\langle u_n|` where :math:`f(\epsilon_n, T_s, \mu)` is the Fermi-Dirac distribution, :math:`\mu` is the chemical potential and :math:`\mathcal{H}_{\mathbf{k}}|u_n\rangle=\epsilon_n|u_n\rangle`. Introducing some smearing is particularly useful when dealing with heterojunctions o inhomogeneous models whose insulating gap is small in order to improve the convergence of the local marker. Default is ``0``, so that no smearing is introduced and a half-filled model is assumed.
            fermidirac_cutoff :
                Cutoff imposed on the Fermi-Dirac distribution to further improve the convergence, mostly when :math:`T_s\neq0`. Default is ``0.1``, which is appropriate in most cases.

        Returns
        -------
            lattice_chern :
                Local Chern marker evaluated on the whole lattice if :python:`direction == None`.
            lcm_direction :
                Local Chern marker evaluated along :python:`direction` starting from :python:`start`.
            gs_projector :
                Ground state projector, returned if :python:`return_projector == True`.
        """

        # Check input variables
        if direction not in [None, 0, 1]:
            raise RuntimeError("Direction allowed are None, 0 (which stands for x), and 1 (which stands for y).")
        
        if direction is not None:
            if direction == 0:
                if start not in range(self.Ly): raise RuntimeError("Invalid start parameter (must be within [0, Ly - 1]).")
            else:
                if start not in range(self.Lx): raise RuntimeError("Invalid start parameter (must be within [0, Lx - 1]).")

        if len(self.r) != 2:
            raise NotImplementedError("The local Chern marker is not yet implemented for dimensionality different than 2.")

        if input_projector is None:
            # Eigenvectors at \Gamma
            eigenvals, eigenvecs = la.eigh(self.hamiltonian)

            # Evaluate the chemical potential
            mu = chemical_potential(eigenvals, smearing_temperature, self.n_occ)
            
            # If smearing_temperature > 0 evaluate the number of states whose occupation is greater than the cutoff
            if smearing_temperature > 1e-6:
                rank = np.sum( fermidirac(eigenvals, smearing_temperature, mu) > fermidirac_cutoff )
            else:
                rank = self.n_occ

            # Build the ground state projector
            gs_projector = contract("ji,ki->jk", smearing(eigenvecs[:, :rank], eigenvecs, eigenvals, smearing_temperature, mu) * eigenvecs[:, :rank], eigenvecs[:, :rank].conjugate())
        else:
            gs_projector = input_projector

        # Chern marker operator
        commut_rx_gsp = self.r[0] @ gs_projector - gs_projector @ self.r[0]
        commut_ry_gsp = self.r[1] @ gs_projector - gs_projector @ self.r[1]
        chern_operator = np.imag(gs_projector @ commut_rx_gsp @ commut_ry_gsp)
        chern_operator *= -4 * np.pi / self.uc_vol

        # If macroscopic_average I have to compute the lattice values with the averages first
        if macroscopic_average or self.disordered:
            lattice_chern = self._average_over_radius(np.diag(chern_operator), cutoff)
        
        if direction is not None:
            # Evaluate index of the selected direction
            indices = self._xy_to_line('x' if direction == 1 else 'y', start)

            # If macroscopic_average consider the averaged lattice, else the Chern operator
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

            # Repeat to ensure that the dimension of the marker matches the dimension of the position matrices, since if not macroscopic_average the value of the marker is defined per unit cell
            lattice_chern = np.repeat(lattice_chern, self.states_uc)

        if not return_projector:
            return np.array(lattice_chern)
        else:
            return np.array(lattice_chern), gs_projector


    def localization_marker(self, direction : int = None, start : int = 0, return_projector : bool = None, input_projector : np.ndarray = None, macroscopic_average : bool = False, cutoff : float = 0.8):
        r"""
        Evaluate the localization marker provided in Ref. `Marrazzo-Resta (2019) <https://doi.org/10.1103/PhysRevLett.122.166602>`_ on the whole lattice if :python:`direction == None`. If :python:`direction` is not :python:`None`, evaluate the localization marker along :python:`direction` starting from :python:`start`. Allowed directions are ``0`` (meaning along :math:`\mathbf{a}_1`), and ``1`` (meaning along :math:`\mathbf{a}_2`).
        
        Parameters
        ----------
            direction :
                Direction along which the localization marker is computed. Default is :python:`None` (returns the marker on the whole lattice). Allowed directions are ``0`` (meaning along :math:`\mathbf{a}_1`), and ``1`` (meaning along :math:`\mathbf{a}_2`).
            start :
                If :python:`direction` is not :python:`None`, indicates the coordinate of the unit cell from which the evaluation of the localization marker starts. For instance, if interested on the value of the local marker along the :math:`\mathbf{a}_1` direction at half height, it should be set :python:`direction = 0` and :python:`start = Ly // 2`.
            return_projector :
                If :python:`True`, returns the ground state projector. Default is :python:`False`.
            input_projector :
                Possibility to provide the ground state projector to be used in the calculation as input. Default is :python:`None`,  which means that the ground state projector is computed from the model stored in the class.
            macroscopic_average :
                If :python:`True`, returns the localization marker averaged in real space over a radius equal to :python:`cutoff`. Default is :python:`False`.
            cutoff :
                Cutoff set for the calculation of the macroscopic average in real space of the localization marker.

        Returns
        --------
            lattice_loc :
                Local Chern marker evaluated on the whole lattice if :python:`direction == None`.
            loc_direction :
                Local Chern marker evaluated along :python:`direction` starting from :python:`start`.
            gs_projector :
                Ground state projector, returned if :python:`return_projector == True`.

        .. note::
            This function is implemented only for TBmodels and PythTB up to now.
        """

        # BEWARE: THIS FUNCTION WORKS ONLY WITH TBMODELS AND PYTHTB UP TO NOW
        if self.model == None:
            raise NotImplementedError("The localization marker is implemented only for TBmodels and PythTB up to now.")

        # Check input variables
        if direction not in [None, 0, 1]:
            raise RuntimeError("Direction allowed are None, 0 (which stands for x), and 1 (which stands for y).")
        
        if direction is not None:
            if direction == 0:
                if start not in range(self.Ly): raise RuntimeError("Invalid start parameter (must be within [0, ny_sites - 1]).")
            else:
                if start not in range(self.Lx): raise RuntimeError("Invalid start parameter (must be within [0, nx_sites - 1]).")

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

        # If macroscopic_average I have to compute the lattice values with the averages first
        if macroscopic_average or self.disordered:
            lattice_loc = self._average_over_radius(np.diag(localization_operator), cutoff)
        
        if direction is not None:
            # Evaluate index of the selected direction
            indices = self._xy_to_line('x' if direction == 1 else 'y', start)

            # If macroscopic_average consider the averaged lattice, else the localization operator
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

            # Repeat to ensure that the dimension of the marker matches the dimension of the position matrices, since if not macroscopic_average the value of the marker is defined per unit cell
            lattice_loc = np.repeat(lattice_loc, self.states_uc)

        if not return_projector:
            return np.array(lattice_loc)
        else:
            return np.array(lattice_loc), gs_projector
        