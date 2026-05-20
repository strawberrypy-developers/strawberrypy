import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from warnings import warn

import tbmodels as tbm
import pythtb as ptb
import wannierberri as wberri

from .config import mpimaster
from .classes import Model
from .postprocessing import average_over_radius

from . import _tbmodels
from . import _pythtb
from . import _wberri

# For backward compatibility with old PythTB versions
ptb_model = ptb.tb_model if ptb.__version__ < "2.0.0" else ptb.tb_model | ptb.TBModel


class FiniteModel(Model):
    r"""A class describing a finite (OBC) model."""

    def __init__(
        self, model: Model, Lx: int = 1, Ly: int = 1, n_occ: int = None, **kwargs
    ):
        r"""Create a finite model of a given :python:`strawberrypy.Model`,
        allowing the calculation of local geometrical and topological markers
        within open boundary conditions.

        Parameters
        ----------
            model :
                A :python:`strawberrypy.Model` instance.
            Lx :
                Number of unit cells repeated along the :math:`\mathbf{a}_1` direction in
                the finite sample.
            Ly :
                Number of unit cells repeated along the :math:`\mathbf{a}_1` direction in
                the finite sample.
            n_occ :
                Number of occupied states. If :python:`None`, this is determined based on
                the original model.
            kwargs :
                Additional keyword arguments of the parent class :python:`Model`.
        """
        # Store the new dimension of the supercell
        self.Lx, self.Ly = Lx, Ly

        # For backward compatibility, will be removed eventually
        if not isinstance(model, Model):
            warn(
                "Passing a 3rd party model directly to the FiniteModel class is deprecated and"
                "will be removed in future versions. Please create a strawberrypy.Model"
                "instance first and then use the appropriate method to create a finite model.",
                DeprecationWarning,
                stacklevel=2,
            )

            from .backends import get_backend

            spinful = kwargs.get("spinful", False)
            is_master_rank = get_backend(nblk=Lx)[0].is_master_rank
            deprecating = True
        else:
            spinful = model.spinful
            is_master_rank = model.backend.is_master_rank
            deprecating = False

        # Create finite model
        if is_master_rank:
            if deprecating:
                self.model = self._make_finite(model)

                if isinstance(model, tbm.Model):
                    st_uc = _tbmodels.calc_states_uc(model)
                elif isinstance(model, ptb_model):
                    st_uc = _pythtb.calc_states_uc(model, spinful)
                elif isinstance(model, wberri.System_R):
                    st_uc = _wberri.calc_states_uc(model)
                else:
                    raise NotImplementedError("Invalid model instance.")
                model.states_uc = st_uc
            else:
                self.model = self._make_finite(model.model)
        else:
            self.model = None

        super().__init__(
            model=self.model,
            spinful=spinful,
            Lx=self.Lx,
            Ly=self.Ly,
            n_occ=n_occ,
            states_uc=model.states_uc,
            disordered=model.disordered,
            has_vacancies=model.has_vacancies,
            is_crystalline=model.is_crystalline,
            **kwargs,
        )

        # If the model comes from TBmodels, making it finite maps positions to reduced coordinates
        if isinstance(model.model, tbm.Model) and is_master_rank:
            self.cart_positions = _tbmodels.get_positions(self.model, self.Lx, self.Ly)
            self.dim_r = self.cart_positions[0, :].size
            self._fix_dimensionality()
            self.r = np.array(
                [sp.csr_array(np.diag(xyz)) for xyz in self.cart_positions.T]
            )

        # Create a shared array per node
        for attr in ["cart_positions"]:
            setattr(self, attr, self.backend.shared_array(getattr(self, attr)))

    def _make_finite(self, model):
        r"""Returns a new instance of a model with open boundary conditions
        (removing periodic hoppings in the Hamiltonian).

        Parameters
        ----------
            model :
                Model instance.
        """
        if isinstance(model, tbm.Model):
            return _tbmodels.make_finite(model, self.Lx, self.Ly)
        elif isinstance(model, ptb_model):
            return _pythtb.make_finite(model, self.Lx, self.Ly)
        elif isinstance(model, wberri.System_R):
            # TODO
            raise NotImplementedError(
                "The function to make finite a WannierBerri model is not yet implemented."
            )
        else:
            raise NotImplementedError("Invalid model instance.")

    #################################################
    # Local markers
    #################################################

    @mpimaster
    def local_chern_marker(
        self,
        direction: int = None,
        start: int = 0,
        return_projector: bool = False,
        input_projector: np.ndarray = None,
        macroscopic_average: bool = False,
        cutoff: float = 0.8,
        bare_marker: bool = False,
        smearing_temperature: float = 0.0,
        fermidirac_cutoff: float = 0.1,
        n_tba: int = 0,
    ):
        r"""
        Evaluate the local Chern marker provided in Ref.\  `Bianco-Resta (2011)
        <https://doi.org/10.1103/PhysRevB.84.241106>`_. This can be evaluated on the whole
        lattice if :python:`direction == None` or along :python:`direction` starting from
        :python:`start`, where directions can be ``0`` (:math:`\mathbf{a}_1` direction),
        or ``1`` (:math:`\mathbf{a}_2` direction).

        Parameters
        ----------
            direction :
                Direction along which the local Chern marker is computed. Default is :python:`None`
                (returns the marker on the whole lattice). Allowed directions are ``0``
                (:math:`\mathbf{a}_1` direction), and ``1`` (:math:`\mathbf{a}_2` direction).
            start :
                If :python:`direction` is not :python:`None`, indicates the coordinate of
                the unit cell from which the evaluation of the local Chern marker starts.
                For instance, if interested on the value of the local marker along the
                :math:`\mathbf{a}_1` direction at half height, it should be set :python:`direction = 0`
                and :python:`start = Ly // 2`.
            return_projector :
                If :python:`True`, returns the ground state projector. Default is :python:`False`.
            input_projector :
                Possibility to provide the ground state projector to be used in the calculation
                as input. Default is :python:`None`, which means that the ground state
                projector is computed from the model.
            macroscopic_average :
                If :python:`True`, returns the local Chern marker averaged in real space
                over a radius equal to :python:`cutoff`. Default is :python:`False`.
            cutoff :
                Cutoff set for the calculation of the macroscopic average in real space of
                the local Chern marker.
            bare_marker :
                If :python:`True` returns the bare local Chern marker, without taking local
                trace and averaging. Default is :python:`False`.
            smearing_temperature :
                Set a fictitious temperature :math:`T_s` to be used when weighting the
                eigenstates of the Hamiltonian comprising the ground state projector. In particular,
                the ground state projector is computed as :math:`\mathcal P=\sum_{n}f(\epsilon_n,
                T_s, \mu)|u_n\rangle\langle u_n|` where :math:`f(\epsilon_n, T_s, \mu)` is the
                Fermi-Dirac distribution, :math:`\mu` is the chemical potential and :math:`\mathcal{H}
                |u_n\rangle=\epsilon_n|u_n\rangle`. Introducing some smearing is particularly
                useful when dealing with heterojunctions or inhomogeneous models whose
                insulating gap is small in order to improve the convergence of the local
                marker. Default is ``0``, so that no smearing is introduced.
            fermidirac_cutoff :
                Cutoff imposed on the Fermi-Dirac distribution to further improve the
                convergence, mostly when :math:`T_s\neq0`. Default is ``0.1``, which is
                appropriate in most cases.
            n_tba :
                Number of states to be added relative to the a priori set filling (defined
                by the :python:`n_occ` parameter or the default half-filling) when applying
                smearing.

        Returns
        -------
            lattice_chern :
                Local Chern marker evaluated on the whole lattice if :python:`direction == None`.
            lcm_direction :
                Local Chern marker evaluated along :python:`direction` starting from :python:`start`.
            gs_projector :
                Ground state projector, returned if :python:`return_projector == True`.
            local_c :
                The bare local Chern marker :math:`C(\mathbf{r})` returned without averaging
                if :python:`bare_marker == True`.
        """

        # Check input variables
        if direction not in [None, 0, 1]:
            raise RuntimeError(
                "Direction allowed are None, 0 (which stands for x), and 1 (which stands for y)."
            )

        if direction is not None:
            if direction == 0:
                if start not in range(self.Ly):
                    raise RuntimeError(
                        "Invalid start parameter (must be within [0, Ly - 1])."
                    )
            else:
                if start not in range(self.Lx):
                    raise RuntimeError(
                        "Invalid start parameter (must be within [0, Lx - 1])."
                    )

        proj_shape = [self.n_orb, self.n_orb]
        if input_projector is None:
            # Distribute the Hamiltonian in block-cyclic layout among MPI ranks if using MPI,
            #   otherwise simply return the global Hamiltonian as the local Hamiltonian
            hamiltonian = self.backend.distribute(self.hamiltonian.toarray())

            # Eigenvectors of the Hamiltonian
            eigenvals, eigenvecs = self.backend.eigh(
                hamiltonian,
                nev=self.n_orb if smearing_temperature > 0 else self.n_occ + n_tba + 1,
                N=self.n_orb,
                collect_evec=False,
                debug_info=self.debug_info,
            )
            del hamiltonian

            if n_tba == 0:
                # Evaluate the chemical potential
                mu = self.physics.chemical_potential(
                    eigenvals, smearing_temperature, self.n_occ
                )

                # If smearing_temperature > 0 evaluate the number of states whose occupation
                #   is greater than the cutoff
                if smearing_temperature > 1e-6:
                    rank_occ = self.physics._get_occupied_states(
                        eigenvals, smearing_temperature, mu, fermidirac_cutoff
                    )

                else:
                    rank_occ = self.n_occ
            else:
                rank_occ, smearing_temperature, mu = self.physics._add_occupied_states(
                    eigenvals, fermidirac_cutoff, self.n_occ, n_tba
                )

            #### ================= Build the ground state projector ================= ###
            coeff = self.physics.smearing(
                eigenvecs,
                eigenvecs,
                eigenvals,
                smearing_temperature,
                mu,
                rank_occ,
                self.n_orb,
                is_dual=False,
            )

            gs_projector = self.physics.get_proj(coeff, eigenvecs, rank_occ, self.n_orb)

        else:
            gs_projector = input_projector

        #### ================= Chern marker operator ================= ###
        r_x = self.backend.distribute_diag(self.cart_positions[:, 0], root=0)
        r_y = self.backend.distribute_diag(self.cart_positions[:, 1], root=0)

        commut_rx_gsp = self.backend.commutator(r_x, gs_projector, proj_shape, proj_shape)
        del r_x
        tmp = self.backend.matmul(gs_projector, commut_rx_gsp, proj_shape, proj_shape)
        del commut_rx_gsp

        commut_ry_gsp = self.backend.commutator(r_y, gs_projector, proj_shape, proj_shape)
        del r_y

        loc_chern_op = self.backend.matmul(tmp, commut_ry_gsp, proj_shape, proj_shape)
        del tmp

        chern_diags = self.backend.get_diag(loc_chern_op, self.n_orb)
        del loc_chern_op

        # StraWBerryPy > 0.3.2 changes the sign for consistency
        chern_diags = (
            4 * np.imag(chern_diags) * np.pi / self.uc_vol
            if self.backend.is_master_rank
            else None
        )

        chern_diags = self.backend.shared_array(chern_diags)

        if bare_marker:
            return chern_diags

        # If macroscopic_average I have to compute the lattice values with the averages first
        if macroscopic_average:
            chern_diags = average_over_radius(self, chern_diags, cutoff, contraction=None)

        if direction is not None:
            lcm_line = self._extract_line_marker(
                chern_diags, direction, start, macroscopic_average
            )

            if not return_projector:
                return lcm_line
            else:
                return lcm_line, gs_projector

        if not macroscopic_average:
            chern_diags = self._trace_unit_cell(chern_diags)

        if not return_projector:
            return chern_diags
        else:
            return chern_diags, gs_projector

    @mpimaster
    def localization_marker(
        self,
        direction: int = None,
        start: int = 0,
        return_projector: bool = None,
        input_projector: np.ndarray = None,
        macroscopic_average: bool = False,
        cutoff: float = 0.8,
        bare_marker: bool = False,
        n_tba: int = 0,
    ):
        r"""
        Evaluate the localization marker provided in Ref.\  `Marrazzo-Resta (2019)
        <https://doi.org/10.1103/PhysRevLett.122.166602>`_. This can be evaluated on the
        whole lattice if :python:`direction == None` or along :python:`direction` starting
        from :python:`start`, where directions can be ``0`` (:math:`\mathbf{a}_1` direction),
        or ``1`` (:math:`\mathbf{a}_2` direction).

        Parameters
        ----------
            direction :
                Direction along which the localization marker is computed. Default is
                :python:`None` (returns the marker on the whole lattice). Allowed directions
                are ``0`` (:math:`\mathbf{a}_1` direction), and ``1`` (:math:`\mathbf{a}_2`
                direction).
            start :
                If :python:`direction` is not :python:`None`, indicates the coordinate of
                the unit cell from which the evaluation of the localization marker starts.
                For instance, if interested on the value of the local marker along the
                :math:`\mathbf{a}_1` direction at half height, it should be set
                :python:`direction = 0` and :python:`start = Ly // 2`.
            return_projector :
                If :python:`True`, returns the ground state projector. Default is :python:`False`.
            input_projector :
                Possibility to provide the ground state projector to be used in the
                calculation as input. Default is :python:`None`,  which means that the
                ground state projector is computed from the model.
            macroscopic_average :
                If :python:`True`, returns the localization marker averaged in real space
                over a radius equal to :python:`cutoff`. Default is :python:`False`.
            cutoff :
                Cutoff set for the calculation of the macroscopic average in real space of
                the localization marker.
            bare_marker :
                If :python:`True` returns the bare localization marker, without taking local
                trace and averaging. Default is :python:`False`.
            n_tba :
                Number of states to be added relative to the a priori set filling (defined
                by the :python:`n_occ` parameter or the default half-filling).

        Returns
        -------
            lattice_loc :
                Localization marker evaluated on the whole lattice if :python:`direction == None`.
            loc_direction :
                Localization marker evaluated along :python:`direction` starting from :python:`start`.
            gs_projector :
                Ground state projector, returned if :python:`return_projector == True`.

        .. note::
            This function is implemented only for TBmodels and PythTB up to now.
        """
        # Check input variables
        if direction not in [None, 0, 1]:
            raise RuntimeError(
                "Direction allowed are None, 0 (which stands for x), and 1 (which stands for y)."
            )

        if direction is not None:
            if direction == 0:
                if start not in range(self.Ly):
                    raise RuntimeError(
                        "Invalid start parameter (must be within [0, Ly - 1])."
                    )
            else:
                if start not in range(self.Lx):
                    raise RuntimeError(
                        "Invalid start parameter (must be within [0, Lx - 1])."
                    )

        proj_shape = [self.n_orb, self.n_orb]
        if input_projector is None:
            # Distribute the Hamiltonian in block-cyclic layout among MPI ranks if using MPI,
            #   otherwise simply return the global Hamiltonian as the local Hamiltonian
            hamiltonian = self.backend.distribute(self.hamiltonian.toarray())

            # Eigenvectors of the Hamiltonian
            _, eigenvecs = self.backend.eigh(
                hamiltonian,
                nev=self.n_occ + n_tba + 1,
                N=self.n_orb,
                collect_evec=False,
                debug_info=self.debug_info,
            )
            del hamiltonian

            # Build the ground state projector
            rank_occ = self.n_occ + n_tba

            gs_projector = self.backend.matmul(
                eigenvecs,
                eigenvecs.conj(),
                proj_shape,
                proj_shape,
                "N",
                "T",
                slc_idx_A=[[0, self.n_orb], [0, rank_occ]],
                slc_idx_B=[[0, self.n_orb], [0, rank_occ]],
            )

        else:
            gs_projector = input_projector

        # Reduced coordinate
        red_positions = self.cart_positions @ la.inv(self.lvecs)

        # Position operator on a square lattice (reduced coordinate adjusted by the dimension of the sample)
        rx = self.backend.distribute_diag(red_positions[:, 0], root=0)
        ry = self.backend.distribute_diag(red_positions[:, 1], root=0)

        # Local marker operator
        commxgsp = self.backend.commutator(rx, gs_projector, proj_shape, proj_shape)
        del rx
        tmp = self.backend.matmul(gs_projector, commxgsp, proj_shape, proj_shape)

        localization_operator = -np.real(
            self.backend.matmul(tmp, commxgsp, proj_shape, proj_shape)
        )
        del tmp, commxgsp

        commygsp = self.backend.commutator(ry, gs_projector, proj_shape, proj_shape)
        del ry
        tmp = self.backend.matmul(gs_projector, commygsp, proj_shape, proj_shape)

        localization_operator -= np.real(
            self.backend.matmul(tmp, commygsp, proj_shape, proj_shape)
        )
        del tmp, commygsp

        localization_diags = self.backend.get_diag(
            localization_operator, N=self.n_orb, root=0
        )
        del localization_operator

        localization_diags = self.backend.shared_array(localization_diags, root=0)

        if bare_marker:
            return localization_diags

        # If macroscopic_average I have to compute the lattice values with the averages first
        if macroscopic_average:
            localization_diags = average_over_radius(
                self, localization_diags, cutoff, contraction=None
            )

        if direction is not None:
            loc_line = self._extract_line_marker(
                localization_diags, direction, start, macroscopic_average
            )

            if not return_projector:
                return loc_line
            else:
                return loc_line, gs_projector

        if not macroscopic_average:
            localization_diags = self._trace_unit_cell(localization_diags)

        if not return_projector:
            return localization_diags
        else:
            return localization_diags, gs_projector

    @mpimaster
    def local_spin_chern_marker(
        self,
        direction: int = None,
        start: int = 0,
        return_projector: bool = None,
        input_projector=None,
        macroscopic_average: bool = False,
        cutoff: float = 0.8,
        bare_marker: bool = False,
        smearing_temperature: float = 0,
        fermidirac_cutoff: float = 0.1,
        n_tba: int = 0,
    ):
        r"""
        Evaluate the local spin-Chern marker provided in Ref.\  `Baù-Marrazzo (2024b)
        <https://doi.org/10.1103/PhysRevB.110.054203>`_. This can be evaluated on the whole
        lattice if :python:`direction == None` or along :python:`direction` starting from
        :python:`start`, where directions can be ``0`` (:math:`\mathbf{a}_1` direction),
        or ``1`` (:math:`\mathbf{a}_2` direction).

        Parameters
        ----------
            direction :
                Direction along which the local spin-Chern marker is computed. Default is
                :python:`None` (returns the marker on the whole lattice). Allowed directions
                are ``0`` (:math:`\mathbf{a}_1` direction), and ``1`` (:math:`\mathbf{a}_2`
                direction).
            start :
                If :python:`direction` is not :python:`None`, indicates the coordinate of
                the unit cell from which the evaluation of the local spin-Chern marker
                starts. For instance, if interested on the value of the local marker along
                the :math:`\mathbf{a}_1` direction at half height, it should be set
                :python:`direction = 0` and :python:`start = Ly // 2`.
            return_projector :
                If :python:`True`, returns the list of projectors :math:`[\mathcal P_+,
                \mathcal P_-]` used in the calculation. Default is :python:`False`.
            input_projector :
                Possibility to provide the list of projectors :math:`[\mathcal P_+,
                \mathcal P_-]` to be used in the calculation as input. Default is
                :python:`None`, which means that they are computed from the model.
            macroscopic_average :
                If :python:`True`, returns the local spin-Chern marker averaged in real
                space over a radius equal to :python:`cutoff`. Default is :python:`False`.
            cutoff :
                Cutoff set for the calculation of the macroscopic average in real space
                of the local spin-Chern marker.
            bare_marker :
                If :python:`True` returns the bare local individual Chern markers, without
                taking local trace and averaging. Default is :python:`False`.
            smearing_temperature :
                Set a fictitious temperature :math:`T_s` to be used when weighting the
                eigenstates of :math:`\mathcal PS_z\mathcal P` comprising the projectors
                :math:`\mathcal P_{\pm}`. In particular, the projectors :math:`\mathcal P_{\pm}`
                are computed as :math:`\mathcal P_{\pm}=\sum_{m:\sigma=\pm}c_m^{\sigma}(T_s)
                |\phi_m^{\sigma}\rangle\langle \phi_m^{\sigma}|` where :math:`|\phi_m^{\sigma}\rangle`
                are the eigenstates of :math:`\mathcal PS_z\mathcal P` with eigenvalue
                :math:`\lambda_m^{\sigma}` whose sign is :math:`\sigma=\pm`. Introducing
                some smearing is particularly useful when dealing with heterojunctions or
                inhomogeneous models whose insulating gap is small in order to improve the
                convergence of the local marker. Default is ``0``, so that no smearing is
                introduced.
            fermidirac_cutoff :
                Cutoff imposed on the Fermi-Dirac distribution to further improve the
                convergence, mostly when :math:`T_s\neq0`. Default is ``0.1``, which is
                appropriate in most cases.
            n_tba :
                Number of states to be added relative to the a priori set filling (defined
                by the :python:`n_occ` parameter or the default half-filling) when applying
                smearing.

        Returns
        -------
            lattice_lscm :
                Local spin-Chern marker evaluated on the whole lattice if :python:`direction == None`.
            lscm_direction :
                Local spin-Chern marker evaluated along :python:`direction` starting from
                :python:`start`.
            gs_projector :
                The list of projectors :math:`[\mathcal P_+, \mathcal P_-]`, returned if
                :python:`return_projector == True`.
            local_cp, local_cm :
                The list of bare local individual Chern markers [:math:`C_+(\mathbf{r})`,
                :math:`C_-(\mathbf{r})`], returned if :python:`bare_marker == True`.
        """

        # Check input variables
        if direction not in [None, 0, 1]:
            raise RuntimeError(
                "Direction allowed are None, 0 (which stands for x), and 1 (which stands for y)"
            )

        if direction is not None:
            if direction == 0:
                if start not in range(self.Ly):
                    raise RuntimeError(
                        "Invalid start parameter (must be within [0, Ly - 1])"
                    )
            else:
                if start not in range(self.Lx):
                    raise RuntimeError(
                        "Invalid start parameter (must be within [0, Lx - 1])"
                    )

        proj_shape = [self.n_orb, self.n_orb]
        if input_projector is None:
            # Distribute the Hamiltonian in block-cyclic layout among MPI ranks if using MPI,
            #   otherwise simply return the global Hamiltonian as the local Hamiltonian
            hamiltonian = self.backend.distribute(self.hamiltonian.toarray())

            eigenvals, canonical_to_H = self.backend.eigh(
                hamiltonian,
                nev=self.n_orb if smearing_temperature > 0 else self.n_occ + n_tba + 1,
                N=self.n_orb,
                collect_evec=False,
                debug_info=self.debug_info,
            )
            del hamiltonian

            if n_tba == 0:
                # Evaluate the chemical potential
                mu = self.physics.chemical_potential(
                    eigenvals, smearing_temperature, self.n_occ, broadcast=True
                )

                # If smearing_temperature > 0 evaluate the number of states whose occupation is greater than the cutoff
                if smearing_temperature > 1e-6:
                    rank_occ = self.physics._get_occupied_states(
                        eigenvals, smearing_temperature, mu, fermidirac_cutoff
                    )

                else:
                    rank_occ = self.n_occ
            else:
                rank_occ, smearing_temperature, mu = self.physics._add_occupied_states(
                    eigenvals, fermidirac_cutoff, self.n_occ, n_tba
                )

            # S^z and PS^zP matrix in the eigenvector basis
            sz = self.backend.distribute(self.sz.toarray(), root=0)
            tmp = self.backend.matmul(
                canonical_to_H.conj(),
                sz,
                proj_shape,
                proj_shape,
                "T",
                "N",
                [[0, self.n_orb], [0, rank_occ]],
            )
            del sz

            pszp = self.backend.matmul(
                tmp,
                canonical_to_H,
                (rank_occ, self.n_orb),
                proj_shape,
                "N",
                "N",
                slc_idx_B=[[0, self.n_orb], [0, rank_occ]],
            )
            del tmp

            _, evecs = self.backend.eigh(
                pszp, nev=rank_occ, N=rank_occ, debug_info=self.debug_info
            )
            del pszp

            evecs = self.backend.matmul(
                canonical_to_H,
                evecs,
                proj_shape,
                (rank_occ, rank_occ),
                "N",
                "N",
                [[0, self.n_orb], [0, rank_occ]],
            )

            # Build the projector onto the lower and higher eigenvalues
            coeff = self.physics.smearing(
                evecs,
                canonical_to_H,
                eigenvals,
                smearing_temperature,
                mu,
                rank_occ,
                self.n_orb,
                is_dual=False,
                spin="down",
            )

            lowerproj = self.physics.get_proj(
                coeff, evecs, rank_occ, self.n_orb, spin="down"
            )
            del coeff

            coeff = self.physics.smearing(
                evecs,
                canonical_to_H,
                eigenvals,
                smearing_temperature,
                mu,
                rank_occ,
                self.n_orb,
                is_dual=False,
                spin="up",
            )

            higherproj = self.physics.get_proj(
                coeff, evecs, rank_occ, self.n_orb, spin="up"
            )
            del coeff

        else:
            higherproj = input_projector[0]
            lowerproj = input_projector[1]

        r_x = self.backend.distribute_diag(self.cart_positions[:, 0], root=0)
        r_y = self.backend.distribute_diag(self.cart_positions[:, 1], root=0)

        ### ================= Positive PSzP eigenvalues - Chern marker operator ================= ###
        commut_rx_gsp = self.backend.commutator(r_x, higherproj, proj_shape, proj_shape)
        tmp = self.backend.matmul(higherproj, commut_rx_gsp, proj_shape, proj_shape)
        del commut_rx_gsp

        commut_ry_gsp = self.backend.commutator(r_y, higherproj, proj_shape, proj_shape)
        chern_operator_plus = self.backend.matmul(
            tmp, commut_ry_gsp, proj_shape, proj_shape
        )
        del tmp, commut_ry_gsp

        # Get the diagonal elements
        local_diags_plus = self.backend.get_diag(
            chern_operator_plus, N=self.n_orb, root=0
        )
        del chern_operator_plus

        ### ================= Negative PSzP eigenvalues - Chern marker operator ================= ###
        commut_rx_gsp = self.backend.commutator(r_x, lowerproj, proj_shape, proj_shape)
        tmp = self.backend.matmul(lowerproj, commut_rx_gsp, proj_shape, proj_shape)
        del commut_rx_gsp

        commut_ry_gsp = self.backend.commutator(r_y, lowerproj, proj_shape, proj_shape)
        chern_operator_minus = self.backend.matmul(
            tmp, commut_ry_gsp, proj_shape, proj_shape
        )
        del tmp, commut_ry_gsp

        # Get the diagonal elements
        local_diags_minus = self.backend.get_diag(
            chern_operator_minus, N=self.n_orb, root=0
        )
        del chern_operator_minus

        local_diags_plus = (
            -4 * np.imag(local_diags_plus) * np.pi / self.uc_vol
            if self.backend.is_master_rank
            else None
        )

        local_diags_minus = (
            -4 * np.imag(local_diags_minus) * np.pi / self.uc_vol
            if self.backend.is_master_rank
            else None
        )

        chern_diags_plus = self.backend.shared_array(local_diags_plus, root=0)
        del local_diags_plus
        chern_diags_minus = self.backend.shared_array(local_diags_minus, root=0)
        del local_diags_minus

        if bare_marker:
            return np.array([chern_diags_plus, chern_diags_minus])

        # If macroscopic average I have to compute the lattice values with the averages first
        if macroscopic_average:
            chern_diags_plus = average_over_radius(
                self, chern_diags_plus, cutoff, contraction=None
            )
            chern_diags_minus = average_over_radius(
                self, chern_diags_minus, cutoff, contraction=None
            )

        if direction is not None:
            lscm_plus = self._extract_line_marker(
                chern_diags_plus, direction, start, macroscopic_average
            )
            lscm_minus = self._extract_line_marker(
                chern_diags_minus, direction, start, macroscopic_average
            )

            if not return_projector:
                return np.abs(np.fmod(0.5 * (lscm_plus - lscm_minus), 2))
            else:
                return np.abs(np.fmod(0.5 * (lscm_plus - lscm_minus), 2)), np.array(
                    [higherproj, lowerproj]
                )

        # If not macroscopic averages I sum the values of the Chern operators of the unit cell
        if not macroscopic_average:
            chern_diags_plus = self._trace_unit_cell(chern_diags_plus)
            chern_diags_minus = self._trace_unit_cell(chern_diags_minus)

        if not return_projector:
            return np.abs(np.fmod(0.5 * (chern_diags_plus - chern_diags_minus), 2))
        else:
            return np.abs(np.fmod(0.5 * (chern_diags_plus - chern_diags_minus), 2)), [
                higherproj,
                lowerproj,
            ]

    @mpimaster
    def local_z2_marker(
        self,
        direction: int = None,
        start: int = 0,
        return_projector: bool = None,
        input_projector=None,
        macroscopic_average: bool = False,
        cutoff: float = 0.8,
        bare_marker: bool = False,
        smearing_temperature: float = 0,
        fermidirac_cutoff: float = 0.1,
        trial_projections=None,
        n_tba: int = 0,
    ):
        r"""
        Evaluate the local :math:`\mathbb{Z}_2` marker provided in Ref.\  `Baù-Marrazzo (2024b)
        <https://doi.org/10.1103/PhysRevB.110.054203>`_. This can be evaluated on the whole
        lattice if :python:`direction == None` or along :python:`direction` starting from
        :python:`start`, where directions can be ``0`` (:math:`\mathbf{a}_1` direction),
        or ``1`` (:math:`\mathbf{a}_2` direction).

        Parameters
        ----------
            direction :
                Direction along which the local :math:`\mathbb{Z}_2` marker is computed.
                Default is :python:`None` (returns the marker on the whole lattice). Allowed
                directions are ``0`` (:math:`\mathbf{a}_1` direction), and ``1``
                (:math:`\mathbf{a}_2` direction).
            start :
                If :python:`direction` is not :python:`None`, indicates the coordinate of
                the unit cell from which the evaluation of the local :math:`\mathbb{Z}_2`
                marker starts. For instance, if interested on the value of the local marker
                along the :math:`\mathbf{a}_1` direction at half height, it should be set
                :python:`direction = 0` and :python:`start = Ly // 2`.
            return_projector :
                If :python:`True`, returns the list of projectors :math:`[\mathcal P_1,
                \Theta\mathcal P_1\Theta^{-1}]` used in the calculation, where :math:`\Theta`
                is the time reversa operator. Default is :python:`False`.
            input_projector :
                Possibility to provide the list of projectors :math:`[\mathcal P_1,
                \Theta\mathcal P_1\Theta^{-1}]` to be used in the calculation as input.
                Default is :python:`None`,  which means that they are computed from the model.
            macroscopic_average :
                If :python:`True`, returns the local :math:`\mathbb{Z}_2` marker averaged
                in real space over a radius equal to :python:`cutoff`. Default is :python:`False`.
            cutoff :
                Cutoff set for the calculation of the macroscopic average in real space of
                the local :math:`\mathbb{Z}_2` marker.
            bare_marker :
                If :python:`True` returns the bare local individual Chern markers, without
                taking local trace and averaging. Default is :python:`False`.
            smearing_temperature :
                Set a fictitious temperature :math:`T_s` to be used when weighting the quasi
                Wannier functions comprising the projector :math:`\mathcal P_{1}`. In
                particular, the projector :math:`\mathcal P_{1}` is computed as :math:`\mathcal
                P_{1}=\sum_{m}c_m(T_s)|w_m\rangle\langle w_m|` where :math:`|w_m\rangle` are
                the quasi-Wannier functions. These are obtained minimizing the spillage
                between :math:`\mathcal P=\sum_n f(\epsilon_n, T_s, \mu)|u_n\rangle\langle u_n|`
                and :math:`\mathcal P_{\Theta}=\mathcal P_1+\Theta\mathcal P_1\Theta^{-1}`,
                where :math:`|u_n\rangle` is the periodic part of the *n*-th Bloch function
                and :math:`f(\epsilon, T, \mu)` is the Fermi-Dirac distribution. Introducing
                some smearing is particularly useful when dealing with heterojunctions or
                inhomogeneous models whose insulating gap is small in order to improve the
                convergence of the local marker. Default is ``0``, so that no smearing is
                introduced.
            fermidirac_cutoff :
                Cutoff imposed on the Fermi-Dirac distribution to further improve the
                convergence, mostly when :math:`T_s\neq0`. Default is ``0.1``, which is
                appropriate in most cases.
            trial_projections :
                The :math:`N\times N` matrix of trial projections to be used when computing
                the quasi-Wannier functions, where :math:`N` is the number of states per
                primitive cell. Default is :python:`None`, meaning a default choice projecting
                on spin-up and spin-down states is employed.
            n_tba :
                Number of states to be added relative to the a priori set filling (defined
                by the :python:`n_occ` parameter or the default half-filling) when applying
                smearing.

        Returns
        -------
            lattice_lz2m :
                Local :math:`\mathbb{Z}_2` marker evaluated on the whole lattice if
                :python:`direction == None`.
            lz2m_direction :
                Local :math:`\mathbb{Z}_2` marker evaluated along :python:`direction`
                starting from :python:`start`.
            gs_projector :
                The list of projectors :math:`[\mathcal P_1, \Theta\mathcal P_1\Theta^{-1}]`,
                returned if :python:`return_projector == True`.
            local_c1, local_c2 :
                The bare local individual Chern markers :math:`C_1(\mathbf{r})` and
                :math:`C_2(\mathbf{r})`, returned if :python:`bare_marker == True`.
        """

        # Check input variables
        if direction not in [None, 0, 1]:
            raise RuntimeError(
                "Direction allowed are None, 0 (which stands for x), and 1 (which stands for y)"
            )

        if direction is not None:
            if direction == 0:
                if start not in range(self.Ly):
                    raise RuntimeError(
                        "Invalid start parameter (must be within [0, Ly - 1])"
                    )
            else:
                if start not in range(self.Lx):
                    raise RuntimeError(
                        "Invalid start parameter (must be within [0, Lx - 1])"
                    )

        proj_shape = [self.n_orb, self.n_orb]
        if input_projector is None:
            # Distribute the Hamiltonian in block-cyclic layout among MPI ranks if using MPI,
            #   otherwise simply return the global Hamiltonian as the local Hamiltonian
            hamiltonian = self.backend.distribute(self.hamiltonian.toarray())

            eigenvals, eigenvecs = self.backend.eigh(
                hamiltonian,
                nev=self.n_orb if smearing_temperature > 0 else self.n_occ + n_tba + 1,
                N=self.n_orb,
                collect_evec=False,
                debug_info=self.debug_info,
            )
            del hamiltonian

            if n_tba == 0:
                # Evaluate the chemical potential
                mu = self.physics.chemical_potential(
                    eigenvals, smearing_temperature, self.n_occ
                )

                # If smearing_temperature > 0 evaluate the number of states whose occupation is greater than the cutoff
                if smearing_temperature > 1e-6:
                    rank_occ = self.physics._get_occupied_states(
                        eigenvals, smearing_temperature, mu, fermidirac_cutoff
                    )

                else:
                    rank_occ = self.n_occ
            else:
                rank_occ, smearing_temperature, mu = self.physics._add_occupied_states(
                    eigenvals, fermidirac_cutoff, self.n_occ, n_tba
                )
            nsub = rank_occ // 2

            # Compute qWFs via projection
            eigenvecs_projected = self.physics._delta_projection(
                eigenvecs, rank_occ, trial_projections, self.states_uc, self.n_orb
            )

            vectors, tr_vectors = self.physics._time_reversal_separation(
                eigenvecs_projected, rank_occ, self.n_orb
            )
            del eigenvecs_projected

            # Compute the two projectors
            coeff = self.physics.smearing(
                np.concatenate((vectors, tr_vectors), axis=1),
                eigenvecs,
                eigenvals,
                smearing_temperature,
                mu,
                nsub,
                self.n_orb,
                is_dual=False,
            )

            projector = self.physics.get_proj(coeff, vectors, nsub, self.n_orb)

            trprojector = self.physics.get_proj(coeff, tr_vectors, nsub, self.n_orb)

        else:
            projector = input_projector[0]
            trprojector = input_projector[1]
            rank_occ = self.n_occ

        r_x = self.backend.distribute_diag(self.cart_positions[:, 0], root=0)
        r_y = self.backend.distribute_diag(self.cart_positions[:, 1], root=0)

        ### ================= Original eigenvalues - Chern marker operator  ================= ###
        commut_rx_gsp = self.backend.commutator(r_x, projector, proj_shape, proj_shape)
        tmp = self.backend.matmul(projector, commut_rx_gsp, proj_shape, proj_shape)
        del commut_rx_gsp
        commut_ry_gsp = self.backend.commutator(r_y, projector, proj_shape, proj_shape)
        chern_operator_1 = self.backend.matmul(tmp, commut_ry_gsp, proj_shape, proj_shape)
        del tmp, commut_ry_gsp

        # Get the diagonal elements
        local_diags_1 = self.backend.get_diag(chern_operator_1, N=self.n_orb, root=0)
        del chern_operator_1

        ### ================= Time-reversed eigenvalues - Chern marker operator  ================= ###
        commut_rx_gsp = self.backend.commutator(r_x, trprojector, proj_shape, proj_shape)
        tmp = self.backend.matmul(trprojector, commut_rx_gsp, proj_shape, proj_shape)
        del commut_rx_gsp
        commut_ry_gsp = self.backend.commutator(r_y, trprojector, proj_shape, proj_shape)
        chern_operator_2 = self.backend.matmul(tmp, commut_ry_gsp, proj_shape, proj_shape)
        del tmp, commut_ry_gsp

        # Get the diagonal elements
        local_diags_2 = self.backend.get_diag(chern_operator_2, N=self.n_orb, root=0)
        del chern_operator_2

        local_diags_1 = (
            -4 * np.imag(local_diags_1) * np.pi / self.uc_vol
            if self.backend.is_master_rank
            else None
        )

        local_diags_2 = (
            -4 * np.imag(local_diags_2) * np.pi / self.uc_vol
            if self.backend.is_master_rank
            else None
        )

        chern_diags_1 = self.backend.shared_array(local_diags_1, root=0)
        del local_diags_1
        chern_diags_2 = self.backend.shared_array(local_diags_2, root=0)
        del local_diags_2

        if bare_marker:
            return np.array([chern_diags_1, chern_diags_2])

        # If macroscopic average I have to compute the lattice values with the averages first
        if macroscopic_average:
            chern_diags_1 = average_over_radius(
                self, chern_diags_1, cutoff, contraction=None
            )
            chern_diags_2 = average_over_radius(
                self, chern_diags_2, cutoff, contraction=None
            )

        if direction is not None:
            lz2m_1 = self._extract_line_marker(
                chern_diags_1, direction, start, macroscopic_average
            )
            lz2m_2 = self._extract_line_marker(
                chern_diags_2, direction, start, macroscopic_average
            )

            if not return_projector:
                return np.abs(np.fmod(0.5 * (lz2m_1 - lz2m_2), 2))
            else:
                return np.abs(np.fmod(0.5 * (lz2m_1 - lz2m_2), 2)), np.array(
                    [projector, trprojector]
                )

        # If not macroscopic averages I sum the values of the Chern operators of the unit cell
        if not macroscopic_average:
            chern_diags_1 = self._trace_unit_cell(chern_diags_1)
            chern_diags_2 = self._trace_unit_cell(chern_diags_2)

        if not return_projector:
            return np.abs(np.fmod(0.5 * (chern_diags_1 - chern_diags_2), 2))
        else:
            return np.abs(np.fmod(0.5 * (chern_diags_1 - chern_diags_2), 2)), [
                projector,
                trprojector,
            ]
