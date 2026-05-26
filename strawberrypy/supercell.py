import numpy as np
from warnings import warn

import tbmodels as tbm
import pythtb as ptb
import wannierberri as wberri

from .config import USE_MPI, mpimaster
from .classes import Model
from .postprocessing import pbc_lattice_contraction, average_over_radius

# For backward compatibility with old PythTB versions
ptb_model = ptb.tb_model if ptb.__version__ < "2.0.0" else ptb.tb_model | ptb.TBModel


class Supercell(Model):
    r"""A class describing a supercell (PBC) of a given model."""

    def __init__(
        self, model: Model, Lx: int = 1, Ly: int = 1, n_occ: int = None, **kwargs
    ):
        r"""Create a supercell of a given :python:`strawberrypy.Model`, allowing
        the calculation of single-point invariants and local geometrical and
        topological markers within periodic boundary conditions.

        Parameters
        ----------
            model :
                A :python:`strawberrypy.Model` instance.
            Lx :
                Number of unit cells repeated along the :math:`\mathbf{a}_1` direction in
                the supercell.
            Ly :
                Number of unit cells repeated along the :math:`\mathbf{a}_2` direction in
                the supercell.
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
                "Passing a 3rd party model directly to the Supercell class is deprecated and"
                "will be removed in future versions. Please create a strawberrypy.Model"
                "instance first and then use the appropriate method to create a Supercell.",
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

        # Create supercell
        if is_master_rank:
            if deprecating:
                self.model = self._make_supercell(model)

                from . import _tbmodels, _pythtb, _wberri

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
                self.model = self._make_supercell(model.model)
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

    def _make_supercell(self, model):
        r"""Create :math:`Lx \times Ly` supercell for a 2D tight-binding model.

        If a 3D tight-binding model is provided, the supercell is created in the *xy*-plane.

        Parameters
        ----------
            model :
                A tight-binding model instance.

        Returns
        -------
            supercell_model :
                A tight-binding model instance of the supercell.
        """
        if self.Lx != 1 or self.Ly != 1:
            if isinstance(model, tbm.Model):
                return (
                    model.supercell([self.Lx, self.Ly])
                    if model.dim == 2
                    else model.supercell([self.Lx, self.Ly, 1])
                )
            elif isinstance(model, ptb_model):
                # For backward compatibility with old PythTB versions
                dimr = model.dim_r if hasattr(model, "dim_r") else model._dim_r

                return (
                    model.make_supercell([[self.Lx, 0], [0, self.Ly]])
                    if dimr == 2
                    else model.make_supercell(
                        [[self.Lx, 0, 0], [0, self.Ly, 0], [0, 0, 1]]
                    )
                )
            elif isinstance(model, wberri.System_R):
                # TODO
                raise NotImplementedError(
                    "Supercell construction for wannierberri models is not implemented yet."
                )
            else:
                raise ValueError("Invalid model instance.")
        else:
            return model

    def _periodic_gauge(self, u_n0, b, glob_shape_u_n0, n_occ=None) -> np.ndarray:
        r"""Returns the matrix of occupied eigenvectors at the edge of the
        Brillouin zone imposing periodic gauge upon the eigenvectors at
        :math:`\Gamma`.

        Parameters
        ----------
            u_n0 :
                Matrix of Hamiltonian eigenstates at :math:`\Gamma`-point.
            :math:`\mathbf{b}` :
                Reciprocal lattice vector indicating the edge of the Brillouin zone.
            glob_shape_u_n0:
                Global shape (rows,cols) of the Hamiltonian eigenstates without any
                slicing or operations to it.
            n_occ :
                Number of occupied states.

        Returns
        -------
            np.ndarray
                Matrix of occupied eigenvectors at the edge of the Brillouin zone
                obtained by imposing periodic gauge upon the eigenvectors at
                :math:`\Gamma`. The shape of the returned matrix is (n_occ, n_orb).
        """
        if n_occ is None:
            n_occ = self.n_occ

        vec_exp_b = np.exp(-1.0j * self.cart_positions @ b).ravel()
        vec_exp_b = self.backend.distribute_diag(vec_exp_b)
        u_nb = self.backend.matmul(
            u_n0,
            vec_exp_b,
            glob_shape_u_n0,
            (self.n_orb, self.n_orb),
            "T",
            "N",
            [[0, self.n_orb], [0, n_occ]],
        )

        return u_nb  # shape: (n_occ, n_orb)

    def _dual_state(
        self, u_n0, u_nb, glob_shape_u_n0, spin=None, n_sub=None, n_occ=None
    ) -> np.ndarray:
        r"""Returns the "dual states" reported in Eq. (8) in Ref. `Ceresoli-Resta (2007)
        <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.76.012405>`_ for the
        calculation of the single-point Chern number and those in Eq. (10) in Ref.
        `Favata-Marrazzo (2023) <https://iopscience.iop.org/article/10.1088/2516-1075/acba6f/meta>`_
        for the calculation of the single-point spin Chern number.

        Parameters
        ----------
            u_n0 :
                Matrix of Hamiltonian eigenstates at :math:`\Gamma`-point.
            u_nb :
                Matrix of Hamiltonian eigenstates at :math:`\mathbf{b}`-point at the
                edge of the Brillouin zone.
            glob_shape_u_n0:
                Global shape (rows,cols) of the Hamiltonian eigenstates without any
                slicing or operations to it.
            spin :
                In case of :python:`spinful` system, indicates which sector of spin
                projected operator spectra is considered in the calculation of the
                single-point spin Chern number. If :python:`spin` is :python:`'up'`
                (:python:`'down'`), eigenstates of spin projected operator with positive
                (negative) eigenvalues are considered.
            n_sub :
                Number of states used in the single-point invariant calculation. If
                :python:`n_sub == None`, the number of occupied eigenstates without
                smearing is used for the calculation of the single-point Chern number
                and the number of eigenstates of spin projected operator with eigenvalues
                of a given sign (:math:`\pm`) is used for the calculation of the
                single-point spin Chern number.
            n_occ:
                Number of occupied states. If :python:`n_occ == None`, the number of
                occupied eigenstates without smearing is used.

        Returns
        -------
            np.ndarray
                Matrix of dual states at the edge of the Brillouin zone
                obtained by imposing periodic gauge upon the eigenvectors at
                :math:`\Gamma`. The shape of the returned matrix is (n_sub, n_orb).
        """
        if n_sub is None:
            n_occ = self.n_occ
            n_sub = n_occ // 2 if self.spinful else n_occ
        else:
            if n_occ is None:
                n_occ = n_sub * 2 if self.spinful else n_sub

        # transpose of the smatrix
        s_mat_b_trans = self.backend.matmul(
            u_nb,
            u_n0.conj(),
            [n_occ, self.n_orb],
            glob_shape_u_n0,
            "N",
            "N",
            slc_idx_B=[[0, self.n_orb], [0, n_occ]],
        )

        if spin == "down" or spin is None:
            udual_nb = self.backend.linsolve(
                s_mat_b_trans,
                u_nb,
                [n_occ, n_occ],
                [n_occ, self.n_orb],
                [[0, n_sub], [0, n_sub]],
                [[0, n_sub], [0, self.n_orb]],
            )

        elif spin == "up":
            udual_nb = self.backend.linsolve(
                s_mat_b_trans,
                u_nb,
                [n_occ, n_occ],
                [n_occ, self.n_orb],
                [[n_sub, 2 * n_sub], [n_sub, 2 * n_sub]],
                [[n_sub, 2 * n_sub], [0, self.n_orb]],
            )

        return udual_nb

    #################################################
    # Single-point invariants
    #################################################

    @mpimaster
    def single_point_chern(self, formula: str = "both", return_ham_gap: bool = False):
        r"""Evaluate the single-point Chern number provided in Ref.\ 
        `Ceresoli-Resta (2007) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.76.012405>`_.

        Parameters
        ----------
            formula :
                Two possible formulas are used for the calculation of the single-point
                Chern number: the :python:`'asymmetric'` one is the one originally derived
                in Ref. `Ceresoli-Resta (2007)` and uses right-hand discrete derivative;
                the :python:`'symmetric'` one uses symmetric derivative centered in
                :math:`\Gamma`-point and converges faster to the exact value with the
                supercell size. Default value is :python:`'both'`, which returns the
                invariants computed with both symmetric and asymmetric formulas.
            return_ham_gap :
                If :python:`True`, returns the value of the gap between the highest
                occupied and the lowest unoccupied eigenstates of the Hamiltonian at
                :math:`\Gamma`-point. Default value is :python:`False`.
                
        Returns
        -------
            chern :
                Dictionary with results from :python:`'asymmetric'` and/or :python:`'symmetric'`
                formula for the single-point Chern number.
            hamiltonian_gap :
                Gap between the lowest unoccupied and the highest occupied eigenstates
                of the Hamiltonian at :math:`\Gamma`-point if :python:`return_ham_gap == True`.
        """
        b1, b2, _ = self.reciprocal_vecs
        proj_shape = [self.n_orb, self.n_orb]

        # Distribute the Hamiltonian in block-cyclic layout among MPI ranks if using MPI,
        #   otherwise simply return the global Hamiltonian as the local Hamiltonian.
        ham = self.backend.distribute(self.hamiltonian.toarray())
        eigvals, u_n0 = self.backend.eigh(
            ham,
            nev=self.n_occ + 1,
            N=self.n_orb,
            collect_evec=False,
            debug_info=self.debug_info,
        )

        hamiltonian_gap = (
            eigvals[self.n_occ] - eigvals[self.n_occ - 1]
            if self.backend.is_master_rank
            else None
        )

        chern = {}

        u_nb1 = self._periodic_gauge(u_n0, b1, proj_shape, n_occ=self.n_occ)

        udual_nb1 = self._dual_state(u_n0, u_nb1, proj_shape, n_sub=self.n_occ)
        del u_nb1

        u_nb2 = self._periodic_gauge(u_n0, b2, proj_shape, n_occ=self.n_occ)
        udual_nb2 = self._dual_state(u_n0, u_nb2, proj_shape, n_sub=self.n_occ)
        del u_nb2

        if formula == "asymmetric" or formula == "both":
            proj = self.backend.matmul(
                udual_nb1.conj(),
                udual_nb2,
                [self.n_occ, self.n_orb],
                [self.n_occ, self.n_orb],
                "N",
                "T",
            )

            chern["asymmetric"] = -np.imag(self.backend.trace(proj, self.n_occ)) / np.pi

        if formula == "symmetric" or formula == "both":
            u_nb1 = self._periodic_gauge(u_n0, -b1, proj_shape, n_occ=self.n_occ)
            udual_nmb1 = self._dual_state(u_n0, u_nb1, proj_shape)
            del u_nb1

            u_nb2 = self._periodic_gauge(u_n0, -b2, proj_shape, n_occ=self.n_occ)
            udual_nmb2 = self._dual_state(u_n0, u_nb2, proj_shape)
            del u_nb2

            proj = self.backend.matmul(
                (udual_nmb1 - udual_nb1).conj(),
                (udual_nmb2 - udual_nb2),
                [self.n_occ, self.n_orb],
                [self.n_occ, self.n_orb],
                "N",
                "T",
            )

            chern["symmetric"] = -np.imag(self.backend.trace(proj, self.n_occ)) / (
                4 * np.pi
            )

        if return_ham_gap:
            return chern, hamiltonian_gap
        else:
            return chern

    @mpimaster
    def single_point_spin_chern(
        self,
        spin: str = "down",
        formula: str = "both",
        return_pszp_gap: bool = False,
        return_ham_gap: bool = False,
    ):
        r"""Evaluate the single-point spin Chern number derived in Ref.\  `Favata-Marrazzo
        (2023) <https://iopscience.iop.org/article/10.1088/2516-1075/acba6f/meta>`_.

        Parameters
        ----------
            spin :
                Indicates which sector of the spin projected operator spectra is considered
                in the calculation of the single-point spin Chern number. If :python:`spin`
                is :python:`'up'` (:python:`'down'`), eigenstates of spin projected operator
                with positive (negative) eigevalues are considered. Default value is :python:`'down'`.
            formula :
                Two possible formulas are used for the calculation of the single-point
                spin Chern number: the :python:`'asymmetric'` one uses right-hand discrete
                derivative; the :python:`'symmetric'` one uses symmetric derivative centered
                in :math:`\Gamma`-point and converges faster to the exact value with the
                supercell size. Default value is :python:`'both'`, which returns the
                invariants computed with both symmetric and asymmetric formulas.
            return_pszp_gap :
                If :python:`True`, returns the value of the gap between positive and negative
                eigenvalues of the spin operator projected onto occupied states at
                :math:`\Gamma`-point. Default value is :python:`False`.
            return_ham_gap :
                If :python:`True`, returns the value of the gap between the lowest unoccupied
                and the highest occupied eigenstates of the Hamiltonian at :math:`\Gamma`-point.
                Default value is :python:`False`.

        Returns
        -------
            spin_chern :
                Dictionary with results from :python:`'asymmetric'` and/or :python:`'symmetric'`
                formula for the single-point spin Chern number.
            pszp_gap :
                Gap between positive and negative eigenvalues of the spin operator projected
                onto occupied states at :math:`\Gamma`-point if :python:`return_pszp_gap == True`.
            hamiltonian_gap :
                Gap between the lowest unoccupied and the highest occupied eigenstates of
                the Hamiltonian at :math:`\Gamma`-point if :python:`return_ham_gap == True`.
        """
        n_sub = self.n_occ // 2

        b1, b2, _ = self.reciprocal_vecs
        proj_shape = [self.n_orb, self.n_orb]

        # Distribute the Hamiltonian in block-cyclic layout among MPI ranks if using MPI,
        #   otherwise simply return the global Hamiltonian as the local Hamiltonian.
        ham = self.backend.distribute(self.hamiltonian.toarray())
        eigvals, u_n0 = self.backend.eigh(
            ham,
            nev=self.n_orb,
            N=self.n_orb,
            collect_evec=False,
            debug_info=self.debug_info,
        )

        hamiltonian_gap = (
            eigvals[self.n_occ] - eigvals[self.n_occ - 1]
            if self.backend.is_master_rank
            else None
        )
        sz = self.backend.distribute(self.sz.toarray())

        tmp = self.backend.matmul(
            u_n0.conj(),
            sz,
            proj_shape,
            proj_shape,
            "T",
            "N",
            [[0, self.n_orb], [0, self.n_occ]],
        )
        del sz

        pszp = self.backend.matmul(
            tmp,
            u_n0,
            (self.n_occ, self.n_orb),
            proj_shape,
            "N",
            "N",
            slc_idx_B=[[0, self.n_orb], [0, self.n_occ]],
        )
        del tmp

        eval_pszp, evec_pszp = self.backend.eigh(
            pszp, self.n_occ, self.n_occ, debug_info=self.debug_info
        )
        del pszp

        spin_chern = {}
        pszp_gap = None
        if self.backend.is_master_rank:
            pszp_gap = eval_pszp[n_sub] - eval_pszp[n_sub - 1]
            if pszp_gap < 10 ** (-14):
                print("Closing P Sz P gap!")
            elif eval_pszp[n_sub] * eval_pszp[n_sub - 1] > 0:
                print("P Sz P spectrum is NOT symmetric!")

        q_0 = self.backend.matmul(
            u_n0,
            evec_pszp,
            proj_shape,
            (self.n_occ, self.n_occ),
            "N",
            "N",
            slc_idx_A=[[0, self.n_orb], [0, self.n_occ]],
        )
        del evec_pszp

        q_b1 = self._periodic_gauge(q_0, b1, [self.n_orb, self.n_occ], n_occ=self.n_occ)
        qdual_b1 = self._dual_state(q_0, q_b1, [self.n_orb, self.n_occ], spin)
        del q_b1

        q_b2 = self._periodic_gauge(q_0, b2, [self.n_orb, self.n_occ], n_occ=self.n_occ)
        qdual_b2 = self._dual_state(q_0, q_b2, [self.n_orb, self.n_occ], spin)
        del q_b2

        if spin.lower() == "down":
            slc_idx_states = [[0, n_sub], [0, self.n_orb]]
        elif spin.lower() == "up":
            slc_idx_states = [[n_sub, 2 * n_sub], [0, self.n_orb]]

            # For scalapack linear solver, the dual state vector shape is preserved. For
            #   spin up (down), only the lower (upper) half of the dual state vector is
            #   used, but the shape is still (n_occ, n_orb) instead of (n_sub, n_orb).
            # This will have a problem when using scipy/numpy linear solvers, which expect
            #   the shape of the vector to be (n_sub, n_orb). To avoid this problem, we
            #   slice the dual state vector to have the shape (n_sub, n_orb) when using
            #   scipy/numpy linear solvers
            if not USE_MPI:
                slc_idx_states = [[0, n_sub], [0, self.n_orb]]

        if formula == "asymmetric" or formula == "both":
            proj = self.backend.matmul(
                qdual_b1.conj(),
                qdual_b2,
                [self.n_occ, self.n_orb],
                [self.n_occ, self.n_orb],
                "N",
                "T",
                slc_idx_states,
                slc_idx_states,
            )
            spin_chern["asymmetric"] = -np.imag(self.backend.trace(proj, n_sub)) / np.pi

        if formula == "symmetric" or formula == "both":
            q_b1 = self._periodic_gauge(
                q_0, -b1, [self.n_orb, self.n_occ], n_occ=self.n_occ
            )
            qdual_mb1 = self._dual_state(q_0, q_b1, [self.n_orb, self.n_occ], spin)
            del q_b1

            q_b2 = self._periodic_gauge(
                q_0, -b2, [self.n_orb, self.n_occ], n_occ=self.n_occ
            )
            qdual_mb2 = self._dual_state(q_0, q_b2, [self.n_orb, self.n_occ], spin)
            del q_b2

            proj = self.backend.matmul(
                (qdual_mb1 - qdual_b1).conj(),
                (qdual_mb2 - qdual_b2),
                [self.n_occ, self.n_orb],
                [self.n_occ, self.n_orb],
                "N",
                "T",
                slc_idx_states,
                slc_idx_states,
            )
            spin_chern["symmetric"] = -np.imag(self.backend.trace(proj, n_sub)) / (
                4 * np.pi
            )

        if return_pszp_gap and return_ham_gap:
            return (spin_chern, pszp_gap, hamiltonian_gap)
        elif return_pszp_gap:
            return spin_chern, pszp_gap
        elif return_ham_gap:
            return spin_chern, hamiltonian_gap
        else:
            return spin_chern

    #################################################
    # Local PBC marker
    #################################################

    @mpimaster
    def pbc_local_chern_marker(
        self,
        direction: int = None,
        start: int = 0,
        return_projector: bool = False,
        input_projector=None,
        formula: str = "symmetric",
        macroscopic_average: bool = False,
        cutoff: float = 0.8,
        bare_marker: bool = False,
        smearing_temperature: float = 0,
        fermidirac_cutoff: float = 0.1,
        n_tba: int = 0,
        until_gap: bool = False,
        gap_tol: float = 1e-8,
    ):
        r"""
        Evaluate the PBC local Chern marker provided in Ref.\  `Baù-Marrazzo (2024a)
        <https://doi.org/10.1103/PhysRevB.109.014206>`_. This can be evaluated on the whole
        lattice if :python:`direction == None` or along :python:`direction` starting from
        :python:`start`, where directions can be ``0`` (:math:`\mathbf{a}_1` direction),
        or ``1`` (:math:`\mathbf{a}_2` direction).

        Parameters
        ----------
            direction :
                Direction along which to compute the PBC local Chern marker. Default is
                :python:`None` (returns the marker on the whole supercell). Allowed
                directions are ``0`` (:math:`\mathbf{a}_1` direction), and ``1``
                (:math:`\mathbf{a}_2` direction).
            start :
                If :python:`direction` is not :python:`None`, is the coordinate of the
                unit cell from which the evaluation of the PBC local Chern marker starts.
                For instance, if interested on the value of the PBC local marker along the
                :math:`\mathbf{a}_1` direction at half height, it should be set
                :python:`direction = 0` and :python:`start = Ly // 2`.
            return_projector :
                If :python:`True`, returns the ground state projector at the end of the
                calculation. Default is :python:`False`.
            input_projector :
                Input the list of projectors :math:`[\mathcal{P}_{\mathbf{b}_1},
                \mathcal{P}_{\mathbf{b}_2},\mathcal{P}_{\Gamma}]` (or :math:`[\mathcal{P}_{\mathbf{b}_1},
                \mathcal{P}_{\mathbf{b}_2},\mathcal{P}_{-\mathbf{b}_1},\mathcal{P}_{-\mathbf{b}_2},
                \mathcal{P}_{\Gamma}]` if :python:`formula == 'symmetric'`) to be used in
                the calculation. Default is :python:`None`, which means that it is computed
                from the model.
            formula :
                Formula to be used. Default is :python:`'symmetric'`, which is computationally
                more demanding but converges faster. Any other input will result in the
                :python:`'asymmetric'` formulation.
            macroscopic_average :
                If :python:`True`, returns the PBC local Chern marker averaged in real
                space over a radius equal to :python:`cutoff`. Default is :python:`False`.
            cutoff :
                Cutoff set for the calculation of the macroscopic average in real space
                of the PBC local Chern marker.
            bare_marker :
                If :python:`True` returns the bare PBC local Chern marker, without taking
                local trace and averaging. Default is :python:`False`.
            smearing_temperature :
                Set a fictitious temperature :math:`T_s` to be used when weighting the
                eigenstates of the Hamiltonian comprising the ground state projector. In
                particular, the ground state projector is computed as :math:`\mathcal
                P=\sum_{n}f(\epsilon_n, T_s, \mu)|u_n\rangle\langle u_n|` where
                :math:`f(\epsilon_n, T_s, \mu)` is the Fermi-Dirac distribution, :math:`\mu`
                is the chemical potential and :math:`\mathcal{H}_{\Gamma}|u_n\rangle=\epsilon_n
                |u_n\rangle`. Introducing some smearing is particularly useful when dealing
                with heterojunctions o inhomogeneous models whose insulating gap is small
                in order to improve the convergence of the local marker. Default is ``0``,
                so no smearing is introduced.
            fermidirac_cutoff :
                Cutoff imposed on the Fermi-Dirac distribution to further improve the
                convergence, mostly when :math:`T_s\neq0`. Default is ``0.1``, which is
                appropriate in most cases.
            n_tba :
                Number of states to be added relative to the a priori set filling (defined
                by the :python:`n_occ` parameter or the default half-filling) when applying
                smearing.
            until_gap :
                If :python:`True`, the minimum number of states in order to have an energy
                gap larger than :python:`gap_tol` is added to :python:`n_occ` in the
                calculation of the marker. Default is :python:`False`.
            gap_tol :
                Minimum value of the gap when adding states to :python:`n_occ` in the
                calculation of the marker if :python:`until_gap == True`. Default is
                python:`1e-8`.

        Returns
        -------
            lattice_chern :
                PBC local Chern marker evaluated on the whole lattice if :python:`direction == None`.
            lcm_direction :
                PBC local Chern marker evaluated along :python:`direction` starting from
                :python:`start`.
            return_proj :
                List of projectors :math:`[\mathcal{P}_{\mathbf{b}_1},\mathcal{P}_{\mathbf{b}_2},
                \mathcal{P}_{\Gamma}]` (or :math:`[\mathcal{P}_{\mathbf{b}_1},
                \mathcal{P}_{\mathbf{b}_2},\mathcal{P}_{-\mathbf{b}_1}, \mathcal{P}_{-\mathbf{b}_2},
                \mathcal{P}_{\Gamma}]` if :python:`formula == 'symmetric'`) used in
                the calculation if :python:`return_projector == True`.
            local_c :
                The bare PBC local Chern marker :math:`C(\mathbf{r})` not averaged,
                returned if :python:`bare_marker == True`.
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

        return_proj = []
        b1, b2, _ = self.reciprocal_vecs
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

            if not until_gap:
                if n_tba == 0:
                    # Find the chemical potential
                    mu = self.physics.chemical_potential(
                        eigenvals, smearing_temperature, self.n_occ
                    )
                    # If smearing_temperature > 0 evaluate the number of states whose
                    #   occupation is greater than the cutoff
                    if smearing_temperature > 1e-6:
                        rank_occ = self.physics._get_occupied_states(
                            eigenvals, smearing_temperature, mu, fermidirac_cutoff
                        )
                    else:
                        rank_occ = self.n_occ
                else:
                    rank_occ, smearing_temperature, mu = (
                        self.physics._add_occupied_states(
                            eigenvals, fermidirac_cutoff, self.n_occ, n_tba
                        )
                    )
            else:
                rank_occ, smearing_temperature, mu = self.physics._add_states_until_gap(
                    eigenvals, self.n_occ, gap_tol, fermidirac_cutoff
                )

            fd = self.physics.fermidirac(
                eigenvals[:rank_occ] if self.backend.is_master_rank else None,
                smearing_temperature,
                mu,
            )
            fd = self.backend.distribute_diag(fd)

            # Dual states at b_1
            udual_b1 = self._periodic_gauge(eigenvecs, b1, proj_shape, n_occ=rank_occ)
            udual_b1 = self._dual_state(eigenvecs, udual_b1, proj_shape, n_sub=rank_occ)
            pb1 = self.physics.get_proj(fd, udual_b1, rank_occ, self.n_orb, is_dual=True)
            if self.backend.is_master_rank and return_projector:
                return_proj += [self.backend.gather(pb1, self.n_orb, self.n_orb, root=0)]
            del udual_b1

            # Dual states at b_2
            udual_b2 = self._periodic_gauge(eigenvecs, b2, proj_shape, n_occ=rank_occ)
            udual_b2 = self._dual_state(eigenvecs, udual_b2, proj_shape, n_sub=rank_occ)
            pb2 = self.physics.get_proj(fd, udual_b2, rank_occ, self.n_orb, is_dual=True)
            if self.backend.is_master_rank and return_projector:
                return_proj += [self.backend.gather(pb2, self.n_orb, self.n_orb, root=0)]
            del udual_b2

            p = self.backend.commutator(pb1, pb2, proj_shape, proj_shape)

            # If I want the symmetric formula I need to do the same also for -b_1 and -b_2
            if formula.lower() == "symmetric":
                udual_b1 = self._periodic_gauge(
                    eigenvecs, -b1, proj_shape, n_occ=rank_occ
                )
                udual_b1 = self._dual_state(
                    eigenvecs, udual_b1, proj_shape, n_sub=rank_occ
                )
                pmb1 = self.physics.get_proj(
                    fd, udual_b1, rank_occ, self.n_orb, is_dual=True
                )
                if self.backend.is_master_rank and return_projector:
                    return_proj += [
                        self.backend.gather(pmb1, self.n_orb, self.n_orb, root=0)
                    ]
                del udual_b1

                p -= self.backend.commutator(pmb1, pb2, proj_shape, proj_shape)
                del pb2

                udual_b2 = self._periodic_gauge(
                    eigenvecs, -b2, proj_shape, n_occ=rank_occ
                )
                udual_b2 = self._dual_state(
                    eigenvecs, udual_b2, proj_shape, n_sub=rank_occ
                )
                pmb2 = self.physics.get_proj(
                    fd, udual_b2, rank_occ, self.n_orb, is_dual=True
                )
                if self.backend.is_master_rank and return_projector:
                    return_proj += [
                        self.backend.gather(pmb2, self.n_orb, self.n_orb, root=0)
                    ]
                del udual_b2

                p -= self.backend.commutator(pb1, pmb2, proj_shape, proj_shape)
                del pb1

                p += self.backend.commutator(pmb1, pmb2, proj_shape, proj_shape)
                del pmb1, pmb2

            gsp = self.physics.get_proj(
                fd, eigenvecs, rank_occ, self.n_orb, is_dual=False
            )

            if self.backend.is_master_rank and return_projector:
                return_proj += [self.backend.gather(gsp, self.n_orb, self.n_orb, root=0)]
        else:
            gsp, pb1, pb2 = None, None, None
            if self.backend.is_master_rank:
                pb1 = input_projector[0]
                pb2 = input_projector[1]
                gsp = input_projector[-1]
            pb1 = self.backend.distribute(pb1)
            pb2 = self.backend.distribute(pb2)
            gsp = self.backend.distribute(gsp)

            p = self.backend.commutator(pb1, pb2, proj_shape, proj_shape)

            if formula == "symmetric":
                pmb1, pmb2 = None, None
                if self.backend.is_master_rank:
                    pmb1 = input_projector[2]
                    pmb2 = input_projector[3]
                pmb1 = self.backend.distribute(pmb1)
                pmb2 = self.backend.distribute(pmb2)

                p += (
                    self.backend.commutator(pmb1, pmb2, proj_shape, proj_shape)
                    - self.backend.commutator(pb1, pmb2, proj_shape, proj_shape)
                    - self.backend.commutator(pmb1, pb2, proj_shape, proj_shape)
                )

        loc_chern_op = self.backend.matmul(p, gsp, proj_shape, proj_shape)
        del p, gsp
        chern_diags = self.backend.get_diag(loc_chern_op, self.n_orb)
        del loc_chern_op

        chern_diags = (
            -np.imag(chern_diags) * (self.Lx * self.Ly) / (np.pi * (8 if formula == "symmetric" else 2))
            if self.backend.is_master_rank
            else None
        )

        chern_diags = self.backend.shared_array(chern_diags)

        if bare_marker:
            return chern_diags

        # If macroscopic_average I have to compute the lattice values with the averages
        #   first (explicit PBC contraction passed)
        if macroscopic_average:
            contraction = pbc_lattice_contraction(self, cutoff)
            chern_diags = average_over_radius(
                self, chern_diags, cutoff, contraction=contraction
            )

        if direction is not None:
            pbclcm_line = self._extract_line_marker(
                chern_diags, direction, start, macroscopic_average
            )

            if not return_projector:
                return pbclcm_line
            else:
                return pbclcm_line, np.asarray(return_proj)

        if not macroscopic_average:
            chern_diags = self._trace_unit_cell(chern_diags)

        if not return_projector:
            return chern_diags
        else:
            return chern_diags, np.asarray(return_proj)

    @mpimaster
    def pbc_local_spin_chern_marker(
        self,
        direction: int = None,
        start: int = 0,
        return_projector: bool = False,
        input_projector=None,
        formula: str = "symmetric",
        macroscopic_average: bool = False,
        cutoff: float = 0.8,
        bare_marker: bool = False,
        smearing_temperature: float = 0,
        fermidirac_cutoff: float = 0.1,
        n_tba: int = 0,
        until_gap: bool = False,
        gap_tol: float = 1e-8,
    ):
        r"""
        Evaluate the PBC local spin-Chern marker provided in Ref.\  `Baù-Marrazzo (2024b)
        <https://doi.org/10.1103/PhysRevB.110.054203>`_. This can be evaluated on the whole
        lattice if :python:`direction == None` or along :python:`direction` starting from
        :python:`start`, where directions can be ``0`` (:math:`\mathbf{a}_1` direction),
        or ``1`` (:math:`\mathbf{a}_2` direction).

        Parameters
        ----------
            direction :
                Direction along which to compute the PBC local spin-Chern marker. Default
                is :python:`None` (returns the marker on the whole supercell). Allowed
                directions are ``0`` (:math:`\mathbf{a}_1` direction), and ``1``
                (:math:`\mathbf{a}_2` direction).
            start :
                If :python:`direction` is not :python:`None`, is the coordinate of the
                unit cell from which the evaluation of the PBC local spin-Chern marker
                starts. For instance, if interested on the value of the PBC local marker
                along the :math:`\mathbf{a}_1` direction at half height, it should be set
                :python:`direction = 0` and :python:`start = Ly // 2`.
            return_projector :
                If :python:`True`, returns the projectors :math:`\mathcal P_{\mathbf b_{1,2}}^{\pm}`
                at the end of the calculation. Default is :python:`False`.
            input_projector :
                List of projectors :math:`[\mathcal{P}_{\mathbf{b}_1}^-,\mathcal{P}_{\mathbf{b}_2}^-,
                \mathcal{P}_{\mathbf{b}_1}^+,\mathcal{P}_{\mathbf{b}_2}^+,\mathcal{P}_{\Gamma}^-,
                \mathcal{P}_{\Gamma}^+]` (or :math:`[\mathcal{P}_{\mathbf{b}_1}^-,
                \mathcal{P}_{\mathbf{b}_2}^-,\mathcal{P}_{\mathbf{b}_1}^+,\mathcal{P}_{\mathbf{b}_2}^+,
                \mathcal{P}_{-\mathbf{b}_1}^-,\mathcal{P}_{-\mathbf{b}_2}^-,\mathcal{P}_{-\mathbf{b}_1}^+,
                \mathcal{P}_{-\mathbf{b}_2}^+,\mathcal{P}_{\Gamma}^-,\mathcal{P}_{\Gamma}^+]`
                if :python:`formula == 'symmetric'`) to be used in the calculation.
                Default is :python:`None`, which means that it is computed from the model.
            formula :
                Formula to be used. Default is :python:`'symmetric'`, which is
                computationally more demanding but converges faster. Any other input will
                result in the :python:`'asymmetric'` formulation.
            macroscopic_average :
                If :python:`True`, returns the PBC local spin-Chern marker averaged in
                real space over a radius equal to :python:`cutoff`. Default is :python:`False`.
            cutoff :
                Cutoff set for the calculation of the macroscopic average in real space of
                the PBC local spin-Chern marker.
            bare_marker :
                If :python:`True` returns the bare PBC local individual Chern markers,
                without taking local trace and averaging. Default is :python:`False`.
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
                by the :python:`n_occ` parameter or the default half-filling) when applying smearing.
            until_gap :
                If :python:`True`, the minimum number of states in order to have an energy
                gap larger than :python:`gap_tol` is added to :python:`n_occ` in the
                calculation of the marker. Default is :python:`False`.
            gap_tol :
                Minimum value of the gap when adding states to :python:`n_occ` in the
                calculation of the marker if :python:`until_gap == True`.  Default is :python:`1e-8`.

        Returns
        -------
            lattice_spinchern :
                PBC local spin-Chern marker evaluated on the whole lattice if :python:`direction == None`.
            lscm_direction :
                PBC local spin-Chern marker evaluated along :python:`direction` starting
                from :python:`start`.
            return_proj :
                List of projectors :math:`[\mathcal{P}_{\mathbf{b}_1}^-,\mathcal{P}_{\mathbf{b}_2}^-,
                \mathcal{P}_{\mathbf{b}_1}^+,\mathcal{P}_{\mathbf{b}_2}^+,\mathcal{P}_{\Gamma}^-,
                \mathcal{P}_{\Gamma}^+]` (or :math:`[\mathcal{P}_{\mathbf{b}_1}^-,\mathcal{P}_{\mathbf{b}_2}^-,
                \mathcal{P}_{-\mathbf{b}_1}^-,\mathcal{P}_{-\mathbf{b}_2}^-,\mathcal{P}_{\mathbf{b}_1}^+,
                \mathcal{P}_{\mathbf{b}_2}^+,\mathcal{P}_{-\mathbf{b}_1}^+,\mathcal{P}_{-\mathbf{b}_2}^+,
                \mathcal{P}_{\Gamma}^-, \mathcal{P}_{\Gamma}^+]` if :python:`formula ==
                'symmetric'`) used in the calculation if :python:`return_projector == True`.
            local_cp, local_cm :
                The list of bare PBC local individual Chern markers :math:`C_+(\mathbf{r})`
                and :math:`C_-(\mathbf{r})`, returned if :python:`bare_marker == True`.
        """  # Check input variables
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

        return_proj = []
        b1, b2, _ = self.reciprocal_vecs
        proj_shape = [self.n_orb, self.n_orb]

        if input_projector is None:
            # Distribute the Hamiltonian in block-cyclic layout among MPI ranks if using MPI,
            #   otherwise simply return the global Hamiltonian as the local Hamiltonian
            hamiltonian = self.backend.distribute(self.hamiltonian.toarray())

            # Eigenvectors of the Hamiltonian
            eigenvals, canonical_to_H = self.backend.eigh(
                hamiltonian,
                nev=self.n_orb if smearing_temperature > 0 else self.n_occ + n_tba + 1,
                N=self.n_orb,
                collect_evec=False,
                debug_info=self.debug_info,
            )
            del hamiltonian

            if not until_gap:
                if n_tba == 0:
                    # Find the chemical potential
                    mu = self.physics.chemical_potential(
                        eigenvals, smearing_temperature, self.n_occ
                    )
                    # If smearing_temperature > 0 evaluate the number of states whose
                    #   occupation is greater than the cutoff
                    if smearing_temperature > 1e-6:
                        rank_occ = self.physics._get_occupied_states(
                            eigenvals, smearing_temperature, mu, fermidirac_cutoff
                        )
                    else:
                        rank_occ = self.n_occ
                else:
                    rank_occ, smearing_temperature, mu = (
                        self.physics._add_occupied_states(
                            eigenvals, fermidirac_cutoff, self.n_occ, n_tba
                        )
                    )
            else:
                rank_occ, smearing_temperature, mu = self.physics._add_states_until_gap(
                    eigenvals, self.n_occ, gap_tol, fermidirac_cutoff
                )
            nsub = rank_occ // 2

            sz = self.backend.distribute(self.sz.toarray())
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
                pszp, rank_occ, rank_occ, debug_info=self.debug_info
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

            ### ================= Negative eigenvalues ================= ###
            evecs_b1 = self._periodic_gauge(
                evecs, b1, [self.n_orb, rank_occ], n_occ=rank_occ
            )
            udual_b1 = self._dual_state(
                evecs, evecs_b1, [self.n_orb, rank_occ], "down", n_sub=nsub
            )
            del evecs_b1

            coeff = self.physics.smearing(
                udual_b1,
                canonical_to_H,
                eigenvals,
                smearing_temperature,
                mu,
                rank_occ,
                self.n_orb,
                is_dual=True,
                spin="down",
            )

            pb1_m = self.physics.get_proj(
                coeff, udual_b1, rank_occ, self.n_orb, is_dual=True, spin="down"
            )
            del coeff
            if self.backend.is_master_rank and return_projector:
                return_proj += [
                    self.backend.gather(pb1_m, self.n_orb, self.n_orb, root=0)
                ]
            del udual_b1

            evecs_b2 = self._periodic_gauge(
                evecs, b2, [self.n_orb, rank_occ], n_occ=rank_occ
            )
            udual_b2 = self._dual_state(
                evecs, evecs_b2, [self.n_orb, rank_occ], "down", n_sub=nsub
            )
            del evecs_b2

            coeff = self.physics.smearing(
                udual_b2,
                canonical_to_H,
                eigenvals,
                smearing_temperature,
                mu,
                rank_occ,
                self.n_orb,
                is_dual=True,
                spin="down",
            )
            pb2_m = self.physics.get_proj(
                coeff, udual_b2, rank_occ, self.n_orb, is_dual=True, spin="down"
            )
            del coeff
            if self.backend.is_master_rank and return_projector:
                return_proj += [
                    self.backend.gather(pb2_m, self.n_orb, self.n_orb, root=0)
                ]
            del udual_b2

            pminus = self.backend.commutator(pb1_m, pb2_m, proj_shape, proj_shape)

            if formula.lower() == "symmetric":
                evecs_b1 = self._periodic_gauge(
                    evecs, -b1, [self.n_orb, rank_occ], n_occ=rank_occ
                )
                udual_b1 = self._dual_state(
                    evecs, evecs_b1, [self.n_orb, rank_occ], "down", n_sub=nsub
                )
                del evecs_b1

                coeff = self.physics.smearing(
                    udual_b1,
                    canonical_to_H,
                    eigenvals,
                    smearing_temperature,
                    mu,
                    rank_occ,
                    self.n_orb,
                    is_dual=True,
                    spin="down",
                )
                pmb1_m = self.physics.get_proj(
                    coeff, udual_b1, rank_occ, self.n_orb, is_dual=True, spin="down"
                )
                if self.backend.is_master_rank and return_projector:
                    return_proj += [
                        self.backend.gather(pmb1_m, self.n_orb, self.n_orb, root=0)
                    ]
                del udual_b1

                evecs_b2 = self._periodic_gauge(
                    evecs, -b2, [self.n_orb, rank_occ], n_occ=rank_occ
                )
                udual_b2 = self._dual_state(
                    evecs, evecs_b2, [self.n_orb, rank_occ], "down", n_sub=nsub
                )
                del evecs_b2

                coeff = self.physics.smearing(
                    udual_b2,
                    canonical_to_H,
                    eigenvals,
                    smearing_temperature,
                    mu,
                    rank_occ,
                    self.n_orb,
                    is_dual=True,
                    spin="down",
                )
                pmb2_m = self.physics.get_proj(
                    coeff, udual_b2, rank_occ, self.n_orb, is_dual=True, spin="down"
                )
                del coeff
                if self.backend.is_master_rank and return_projector:
                    return_proj += [
                        self.backend.gather(pmb2_m, self.n_orb, self.n_orb, root=0)
                    ]
                del udual_b2

                pminus -= self.backend.commutator(pb1_m, pmb2_m, proj_shape, proj_shape)
                del pb1_m

                pminus -= self.backend.commutator(pmb1_m, pb2_m, proj_shape, proj_shape)
                del pb2_m

                pminus += self.backend.commutator(pmb1_m, pmb2_m, proj_shape, proj_shape)

            ### ================= Positive eigenvalues ================= ###
            evecs_b1 = self._periodic_gauge(
                evecs, b1, [self.n_orb, rank_occ], n_occ=rank_occ
            )
            udual_b1 = self._dual_state(
                evecs, evecs_b1, [self.n_orb, rank_occ], "up", n_sub=nsub
            )
            del evecs_b1

            coeff = self.physics.smearing(
                udual_b1,
                canonical_to_H,
                eigenvals,
                smearing_temperature,
                mu,
                rank_occ,
                self.n_orb,
                is_dual=True,
                spin="up",
            )
            pb1_p = self.physics.get_proj(
                coeff, udual_b1, rank_occ, self.n_orb, is_dual=True, spin="up"
            )
            del coeff

            if self.backend.is_master_rank and return_projector:
                return_proj += [
                    self.backend.gather(pb1_p, self.n_orb, self.n_orb, root=0)
                ]
            del udual_b1

            evecs_b2 = self._periodic_gauge(
                evecs, b2, [self.n_orb, rank_occ], n_occ=rank_occ
            )
            udual_b2 = self._dual_state(
                evecs, evecs_b2, [self.n_orb, rank_occ], "up", n_sub=nsub
            )
            del evecs_b2

            coeff = self.physics.smearing(
                udual_b2,
                canonical_to_H,
                eigenvals,
                smearing_temperature,
                mu,
                rank_occ,
                self.n_orb,
                is_dual=True,
                spin="up",
            )
            pb2_p = self.physics.get_proj(
                coeff, udual_b2, rank_occ, self.n_orb, is_dual=True, spin="up"
            )
            del coeff
            if self.backend.is_master_rank and return_projector:
                return_proj += [
                    self.backend.gather(pb2_p, self.n_orb, self.n_orb, root=0)
                ]
            del udual_b2

            pplus = self.backend.commutator(pb1_p, pb2_p, proj_shape, proj_shape)

            if formula.lower() == "symmetric":
                evecs_b1 = self._periodic_gauge(
                    evecs, -b1, [self.n_orb, rank_occ], n_occ=rank_occ
                )
                udual_b1 = self._dual_state(
                    evecs, evecs_b1, [self.n_orb, rank_occ], "up", n_sub=nsub
                )
                del evecs_b1

                coeff = self.physics.smearing(
                    udual_b1,
                    canonical_to_H,
                    eigenvals,
                    smearing_temperature,
                    mu,
                    rank_occ,
                    self.n_orb,
                    is_dual=True,
                    spin="up",
                )
                pmb1_p = self.physics.get_proj(
                    coeff, udual_b1, rank_occ, self.n_orb, is_dual=True, spin="up"
                )
                del coeff
                if self.backend.is_master_rank and return_projector:
                    return_proj += [
                        self.backend.gather(pmb1_p, self.n_orb, self.n_orb, root=0)
                    ]
                del udual_b1

                evecs_b2 = self._periodic_gauge(
                    evecs, -b2, [self.n_orb, rank_occ], n_occ=rank_occ
                )
                udual_b2 = self._dual_state(
                    evecs, evecs_b2, [self.n_orb, rank_occ], "up", n_sub=nsub
                )
                del evecs_b2

                coeff = self.physics.smearing(
                    udual_b2,
                    canonical_to_H,
                    eigenvals,
                    smearing_temperature,
                    mu,
                    rank_occ,
                    self.n_orb,
                    is_dual=True,
                    spin="up",
                )
                pmb2_p = self.physics.get_proj(
                    coeff, udual_b2, rank_occ, self.n_orb, is_dual=True, spin="up"
                )
                del coeff
                if self.backend.is_master_rank and return_projector:
                    return_proj += [
                        self.backend.gather(pmb2_p, self.n_orb, self.n_orb, root=0)
                    ]
                del udual_b2

                pplus -= self.backend.commutator(pb1_p, pmb2_p, proj_shape, proj_shape)
                del pb1_p

                pplus -= self.backend.commutator(pmb1_p, pb2_p, proj_shape, proj_shape)
                del pb2_p

                pplus += self.backend.commutator(pmb1_p, pmb2_p, proj_shape, proj_shape)

        else:
            gsp_m, gsp_p, pb1_m, pb2_m, pb1_p, pb2_p = (
                None,
                None,
                None,
                None,
                None,
                None,
            )
            if self.backend.is_master_rank:
                pb1_m = input_projector[0]
                pb2_m = input_projector[1]
                pb1_p = input_projector[2]
                pb2_p = input_projector[3]
                gsp_m = input_projector[-2]
                gsp_p = input_projector[-1]

            pb1_m = self.backend.distribute(pb1_m)
            pb2_m = self.backend.distribute(pb2_m)
            pb1_p = self.backend.distribute(pb1_p)
            pb2_p = self.backend.distribute(pb2_p)
            gsp_m = self.backend.distribute(gsp_m)
            gsp_p = self.backend.distribute(gsp_p)

            pminus = self.backend.commutator(pb1_m, pb2_m, proj_shape, proj_shape)
            pplus = self.backend.commutator(pb1_p, pb2_p, proj_shape, proj_shape)

            if formula == "symmetric":
                pmb1_m, pmb2_m, pmb1_p, pmb2_p = None, None, None, None
                if self.backend.is_master_rank:
                    pmb1_m = input_projector[4]
                    pmb2_m = input_projector[5]
                    pmb1_p = input_projector[6]
                    pmb2_p = input_projector[7]

                pmb1_m = self.backend.distribute(pmb1_m)
                pmb2_m = self.backend.distribute(pmb2_m)
                pmb1_p = self.backend.distribute(pmb1_p)
                pmb2_p = self.backend.distribute(pmb2_p)

                pminus += (
                    self.backend.commutator(pmb1_m, pmb2_m, proj_shape, proj_shape)
                    - self.backend.commutator(pb1_m, pmb2_m, proj_shape, proj_shape)
                    - self.backend.commutator(pmb1_m, pb2_m, proj_shape, proj_shape)
                )

                pplus += (
                    self.backend.commutator(pmb1_p, pmb2_p, proj_shape, proj_shape)
                    - self.backend.commutator(pb1_p, pmb2_p, proj_shape, proj_shape)
                    - self.backend.commutator(pmb1_p, pb2_p, proj_shape, proj_shape)
                )

        ### ================= PBC local spin-Chern markers ================= ###
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
        gsp_m = self.physics.get_proj(
            coeff, evecs, rank_occ, self.n_orb, is_dual=False, spin="down"
        )
        del coeff
        loc_spinchern_op_minus = self.backend.matmul(
            pminus, gsp_m, proj_shape, proj_shape
        )
        if self.backend.is_master_rank and return_projector:
            return_proj += [self.backend.gather(gsp_m, self.n_orb, self.n_orb, root=0)]
        del pminus, gsp_m

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
        gsp_p = self.physics.get_proj(
            coeff, evecs, rank_occ, self.n_orb, is_dual=False, spin="up"
        )
        del coeff
        loc_spinchern_op_plus = self.backend.matmul(pplus, gsp_p, proj_shape, proj_shape)

        if self.backend.is_master_rank and return_projector:
            return_proj += [self.backend.gather(gsp_p, self.n_orb, self.n_orb, root=0)]
        del pplus, gsp_p

        # Get the diagonal elements
        chern_diags_minus = self.backend.get_diag(loc_spinchern_op_minus, N=self.n_orb)
        del loc_spinchern_op_minus

        chern_diags_plus = self.backend.get_diag(loc_spinchern_op_plus, N=self.n_orb)
        del loc_spinchern_op_plus

        chern_diags_minus = (
            -np.imag(chern_diags_minus) * (self.Lx * self.Ly) / (np.pi * (8 if formula == "symmetric" else 2))
            if self.backend.is_master_rank
            else None
        )

        chern_diags_plus = (
            -np.imag(chern_diags_plus) * (self.Lx * self.Ly) / (np.pi * (8 if formula == "symmetric" else 2))
            if self.backend.is_master_rank
            else None
        )

        chern_diags_minus = self.backend.shared_array(chern_diags_minus)
        chern_diags_plus = self.backend.shared_array(chern_diags_plus)

        if bare_marker:
            return np.array([chern_diags_plus, chern_diags_minus])

        # If macroscopic_average I have to compute the lattice values with the averages
        #   first (explicit PBC contraction passed)
        if macroscopic_average:
            contraction = pbc_lattice_contraction(self, cutoff)
            chern_diags_plus = average_over_radius(
                self, chern_diags_plus, cutoff, contraction=contraction
            )
            chern_diags_minus = average_over_radius(
                self, chern_diags_minus, cutoff, contraction=contraction
            )

        if direction is not None:
            pbclcm_plus = self._extract_line_marker(
                chern_diags_plus, direction, start, macroscopic_average
            )
            pbclcm_minus = self._extract_line_marker(
                chern_diags_minus, direction, start, macroscopic_average
            )

            if not return_projector:
                return np.abs(np.fmod(0.5 * (pbclcm_plus - pbclcm_minus), 2))
            else:
                return np.abs(np.fmod(0.5 * (pbclcm_plus - pbclcm_minus), 2)), np.asarray(
                    return_proj
                )

        # If not macroscopic averages I sum the values of the Chern operators of the unit cell
        if not macroscopic_average:
            chern_diags_minus = self._trace_unit_cell(chern_diags_minus)
            chern_diags_plus = self._trace_unit_cell(chern_diags_plus)

        if not return_projector:
            return np.abs(np.fmod(0.5 * (chern_diags_plus - chern_diags_minus), 2))
        else:
            return np.abs(
                np.fmod(0.5 * (chern_diags_plus - chern_diags_minus), 2)
            ), np.array(return_proj)

    @mpimaster
    def pbc_local_z2_marker(
        self,
        direction: int = None,
        start: int = 0,
        return_projector: bool = False,
        input_projector=None,
        formula: str = "symmetric",
        macroscopic_average: bool = False,
        cutoff: float = 0.8,
        bare_marker: bool = False,
        smearing_temperature: float = 0,
        fermidirac_cutoff: float = 0.1,
        trial_projections=None,
    ):
        r"""
        Evaluate the local PBC :math:`\mathbb{Z}_2` marker provided in Ref.\  `Baù-Marrazzo
        (2024b) <https://doi.org/10.1103/PhysRevB.110.054203>`_. This can be evaluated on
        the whole lattice if :python:`direction == None` or along :python:`direction`
        starting from :python:`start`, where directions can be ``0`` (:math:`\mathbf{a}_1`
        direction), or ``1`` (:math:`\mathbf{a}_2` direction).

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
                If :python:`True`, returns the list of projectors :math:`[\mathcal{P}_{1,\Gamma},
                \Theta\mathcal{P}_{1,\Gamma}\Theta^{-1}, \mathcal{P}_{1,\mathbf{b}_1},
                \mathcal{P}_{1,\mathbf{b}_2}, \Theta\mathcal{P}_{1,\mathbf{b}_1}\Theta^{-1},
                \Theta\mathcal{P}_{1,\mathbf{b}_2}\Theta^{-1}]` (or :math:`[\mathcal{P}_{1,\Gamma},
                \Theta\mathcal{P}_{1,\Gamma}\Theta^{-1}, \mathcal{P}_{1,\mathbf{b}_1},
                \mathcal{P}_{1,\mathbf{b}_2}, \Theta\mathcal{P}_{1,\mathbf{b}_1}\Theta^{-1},
                \Theta\mathcal{P}_{1,\mathbf{b}_2}\Theta^{-1}, \mathcal{P}_{1,-\mathbf{b}_1},
                \mathcal{P}_{1, -\mathbf{b}_2}, \Theta\mathcal{P}_{1, -\mathbf{b}_1}\Theta^{-1},
                \Theta\mathcal{P}_{1, -\mathbf{b}_2}\Theta^{-1}]` if :python:`formula ==
                'symmetric'`) used in the calculation, where :math:`\Theta` is the time
                reversal operator. Default is :python:`False`.
            input_projector :
                List of projectors :math:`[\mathcal{P}_{1,\Gamma}, \Theta\mathcal{P}_{1,\Gamma}\Theta^{-1},
                \mathcal{P}_{1,\mathbf{b}_1}, \mathcal{P}_{1,\mathbf{b}_2}, \Theta\mathcal{P}_{1,\mathbf{b}_1}\Theta^{-1},
                \Theta\mathcal{P}_{1,\mathbf{b}_2}\Theta^{-1}]` (or :math:`[\mathcal{P}_{1,\Gamma},
                \Theta\mathcal{P}_{1,\Gamma}\Theta^{-1}, \mathcal{P}_{1,\mathbf{b}_1},
                \mathcal{P}_{1,\mathbf{b}_2}, \Theta\mathcal{P}_{1,\mathbf{b}_1}\Theta^{-1},
                \Theta\mathcal{P}_{1,\mathbf{b}_2}\Theta^{-1}, \mathcal{P}_{1,-\mathbf{b}_1},
                \mathcal{P}_{1, -\mathbf{b}_2}, \Theta\mathcal{P}_{1, -\mathbf{b}_1}\Theta^{-1},
                \Theta\mathcal{P}_{1, -\mathbf{b}_2}\Theta^{-1}]` if :python:`formula == 'symmetric'`)
                to be used in the calculation. Default is :python:`None`, which means that
                it is computed from the model.
            formula :
                Formula to be used. Default is :python:`'symmetric'`, which is computationally
                more demanding but converges faster. Any other input will result in the
                :python:`'asymmetric'` formulation.
            macroscopic_average :
                If :python:`True`, returns the local :math:`\mathbb{Z}_2` marker averaged
                in real space over a radius equal to :python:`cutoff`. Default is :python:`False`.
            cutoff :
                Cutoff set for the calculation of the macroscopic average in real space of
                the local :math:`\mathbb{Z}_2` marker.
            bare_marker :
                If :python:`True` returns the bare PBC local individual Chern markers,
                without taking local trace and averaging. Default is :python:`False`.
            smearing_temperature :
                Set a fictitious temperature :math:`T_s` to be used when weighting the quasi
                Wannier functions comprising the projector :math:`\mathcal P_{1}`. In particular,
                the projector :math:`\mathcal P_{1}` is computed as
                :math:`\mathcal P_{1}=\sum_{m}c_m(T_s)|w_m\rangle\langle w_m|` where
                :math:`|w_m\rangle` are the quasi Wannier functions. These are obtained
                minimizing the spillage between :math:`\mathcal P=\sum_n f(\epsilon_n, T_s, \mu)
                |u_n\rangle\langle u_n|` and :math:`\mathcal P_{\Theta}=\mathcal P_1+\Theta\mathcal P_1\Theta^{-1}`,
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

        Returns
        -------
            lattice_lz2m :
                Local :math:`\mathbb{Z}_2` marker evaluated on the whole lattice if
                :python:`direction == None`.
            lz2m_direction :
                Local :math:`\mathbb{Z}_2` marker evaluated along :python:`direction`
                starting from :python:`start`.
            return_proj :
                The list of projectors :math:`[\mathcal P_1, \Theta\mathcal P_1\Theta^{-1}]`,
                returned if :python:`return_projector == True`.
            local_c1, local_c2 :
                The bare PBC local individual Chern markers :math:`C_1(\mathbf{r})` and
                :math:`C_2(\mathbf{r})`, returned if :python:`bare_marker == True`.

        .. note::
            The local :math:`\mathbb{Z}_2` marker does not currently support the
            :python:`until_gap` and :python:`n_tba` options as the other PBC local
            topological markers and it will updated in the future.
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

        return_proj = []
        b1, b2, _ = self.reciprocal_vecs
        proj_shape = [self.n_orb, self.n_orb]

        if input_projector is None:
            # Distribute the Hamiltonian in block-cyclic layout among MPI ranks if using
            #   MPI, otherwise simply return the global Hamiltonian as the local
            #   Hamiltonian
            hamiltonian = self.backend.distribute(self.hamiltonian.toarray())

            # Eigenvectors of the Hamiltonian
            eigenvals, eigenvecs = self.backend.eigh(
                hamiltonian,
                nev=self.n_orb if smearing_temperature > 0 else self.n_occ + 2,
                N=self.n_orb,
                collect_evec=False,
                debug_info=self.debug_info,
            )
            del hamiltonian

            # Find the chemical potential
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
            nsub = rank_occ // 2

            # Compute qWFs via projection
            eigenvecs_projected = self.physics._delta_projection(
                eigenvecs, rank_occ, trial_projections, self.states_uc, self.n_orb
            )

            vectors, tr_vectors = self.physics._time_reversal_separation(
                eigenvecs_projected, rank_occ, self.n_orb
            )

            ########### ======= Subspace spanned by 'vectors' ========###########
            evecs_b1 = self._periodic_gauge(vectors, b1, [self.n_orb, nsub], n_occ=nsub)
            udual_b1 = self._dual_state(
                vectors, evecs_b1, [self.n_orb, nsub], n_sub=nsub, n_occ=nsub
            )
            del evecs_b1

            coeff = self.physics.smearing(
                udual_b1,
                eigenvecs,
                eigenvals,
                smearing_temperature,
                mu,
                nsub,
                self.n_orb,
                is_dual=True,
                spin=None,
            )
            pb1_1 = self.physics.get_proj(coeff, udual_b1, nsub, self.n_orb, is_dual=True)
            del coeff, udual_b1

            evecs_b2 = self._periodic_gauge(vectors, b2, [self.n_orb, nsub], n_occ=nsub)
            udual_b2 = self._dual_state(
                vectors, evecs_b2, [self.n_orb, nsub], n_sub=nsub, n_occ=nsub
            )
            del evecs_b2
            coeff = self.physics.smearing(
                udual_b2,
                eigenvecs,
                eigenvals,
                smearing_temperature,
                mu,
                nsub,
                self.n_orb,
                is_dual=True,
                spin=None,
            )

            pb2_1 = self.physics.get_proj(coeff, udual_b2, nsub, self.n_orb, is_dual=True)
            del coeff, udual_b2

            p1 = self.backend.commutator(pb1_1, pb2_1, proj_shape, proj_shape)
            if formula.lower() == "symmetric":
                evecs_b1 = self._periodic_gauge(
                    vectors, -b1, [self.n_orb, nsub], n_occ=nsub
                )
                udual_b1 = self._dual_state(
                    vectors, evecs_b1, [self.n_orb, nsub], n_sub=nsub, n_occ=nsub
                )
                del evecs_b1

                coeff = self.physics.smearing(
                    udual_b1,
                    eigenvecs,
                    eigenvals,
                    smearing_temperature,
                    mu,
                    nsub,
                    self.n_orb,
                    is_dual=True,
                    spin=None,
                )
                pmb1_1 = self.physics.get_proj(
                    coeff, udual_b1, nsub, self.n_orb, is_dual=True
                )
                del coeff, udual_b1

                evecs_b2 = self._periodic_gauge(
                    vectors, -b2, [self.n_orb, nsub], n_occ=nsub
                )
                udual_b2 = self._dual_state(
                    vectors, evecs_b2, [self.n_orb, nsub], n_sub=nsub, n_occ=nsub
                )
                del evecs_b2

                coeff = self.physics.smearing(
                    udual_b2,
                    eigenvecs,
                    eigenvals,
                    smearing_temperature,
                    mu,
                    nsub,
                    self.n_orb,
                    is_dual=True,
                    spin=None,
                )
                pmb2_1 = self.physics.get_proj(
                    coeff, udual_b2, nsub, self.n_orb, is_dual=True
                )
                del coeff, udual_b2

                p1 -= self.backend.commutator(pb1_1, pmb2_1, proj_shape, proj_shape)
                del pb1_1
                p1 -= self.backend.commutator(pmb1_1, pb2_1, proj_shape, proj_shape)
                del pb2_1
                p1 += self.backend.commutator(pmb1_1, pmb2_1, proj_shape, proj_shape)
                del pmb1_1, pmb2_1

            ########### ======= Subspace spanned by 'tr_vectors' ========###########
            evecs_b1 = self._periodic_gauge(
                tr_vectors, b1, [self.n_orb, nsub], n_occ=nsub
            )
            udual_b1 = self._dual_state(
                tr_vectors, evecs_b1, [self.n_orb, nsub], n_sub=nsub, n_occ=nsub
            )
            del evecs_b1

            coeff = self.physics.smearing(
                udual_b1,
                eigenvecs,
                eigenvals,
                smearing_temperature,
                mu,
                nsub,
                self.n_orb,
                is_dual=True,
                spin=None,
            )
            pb1_2 = self.physics.get_proj(coeff, udual_b1, nsub, self.n_orb, is_dual=True)
            del coeff, udual_b1

            evecs_b2 = self._periodic_gauge(
                tr_vectors, b2, [self.n_orb, nsub], n_occ=nsub
            )
            udual_b2 = self._dual_state(
                tr_vectors, evecs_b2, [self.n_orb, nsub], n_sub=nsub, n_occ=nsub
            )
            del evecs_b2
            coeff = self.physics.smearing(
                udual_b2,
                eigenvecs,
                eigenvals,
                smearing_temperature,
                mu,
                nsub,
                self.n_orb,
                is_dual=True,
                spin=None,
            )
            pb2_2 = self.physics.get_proj(coeff, udual_b2, nsub, self.n_orb, is_dual=True)
            del coeff, udual_b2

            p2 = self.backend.commutator(pb1_2, pb2_2, proj_shape, proj_shape)
            if formula.lower() == "symmetric":
                evecs_b1 = self._periodic_gauge(
                    tr_vectors, -b1, [self.n_orb, nsub], n_occ=nsub
                )
                udual_b1 = self._dual_state(
                    tr_vectors, evecs_b1, [self.n_orb, nsub], n_sub=nsub, n_occ=nsub
                )
                del evecs_b1
                coeff = self.physics.smearing(
                    udual_b1,
                    eigenvecs,
                    eigenvals,
                    smearing_temperature,
                    mu,
                    nsub,
                    self.n_orb,
                    is_dual=True,
                    spin=None,
                )
                pmb1_2 = self.physics.get_proj(
                    coeff, udual_b1, nsub, self.n_orb, is_dual=True
                )
                del coeff, udual_b1

                evecs_b2 = self._periodic_gauge(
                    tr_vectors, -b2, [self.n_orb, nsub], n_occ=nsub
                )
                udual_b2 = self._dual_state(
                    tr_vectors, evecs_b2, [self.n_orb, nsub], n_sub=nsub, n_occ=nsub
                )
                del evecs_b2

                coeff = self.physics.smearing(
                    udual_b2,
                    eigenvecs,
                    eigenvals,
                    smearing_temperature,
                    mu,
                    nsub,
                    self.n_orb,
                    is_dual=True,
                    spin=None,
                )
                pmb2_2 = self.physics.get_proj(
                    coeff, udual_b2, nsub, self.n_orb, is_dual=True
                )
                del coeff, udual_b2

                p2 -= self.backend.commutator(pb1_2, pmb2_2, proj_shape, proj_shape)
                del pb1_2
                p2 -= self.backend.commutator(pmb1_2, pb2_2, proj_shape, proj_shape)
                del pb2_2
                p2 += self.backend.commutator(pmb1_2, pmb2_2, proj_shape, proj_shape)
                del pmb1_2, pmb2_2
        else:
            gsp_1 = input_projector[0]
            gsp_2 = input_projector[1]

            pb1_1 = input_projector[2]
            pb2_1 = input_projector[3]
            p1 = pb1_1 @ pb2_1 - pb2_1 @ pb1_1

            pb1_2 = input_projector[4]
            pb2_2 = input_projector[5]
            p2 = pb1_2 @ pb2_2 - pb2_2 @ pb1_2

            if formula == "symmetric":
                pmb1_1 = input_projector[6]
                pmb2_1 = input_projector[7]
                p1 += (
                    (pmb1_1 @ pmb2_1 - pmb2_1 @ pmb1_1)
                    - (pb1_1 @ pmb2_1 - pmb2_1 @ pb1_1)
                    - (pmb1_1 @ pb2_1 - pb2_1 @ pmb1_1)
                )

                pmb1_2 = input_projector[8]
                pmb2_2 = input_projector[9]
                p2 += (
                    (pmb1_2 @ pmb2_2 - pmb2_2 @ pmb1_2)
                    - (pb1_2 @ pmb2_2 - pmb2_2 @ pb1_2)
                    - (pmb1_2 @ pb2_2 - pb2_2 @ pmb1_2)
                )

        ########### ======= PBC local Z2 markers ========###########
        coeff = self.physics.smearing(
            vectors,
            eigenvecs,
            eigenvals,
            smearing_temperature,
            mu,
            nsub,
            self.n_orb,
            is_dual=False,
            spin=None,
        )

        gsp_1 = self.physics.get_proj(coeff, vectors, nsub, self.n_orb, is_dual=False)
        del coeff, vectors
        loc_z2_op_1 = self.backend.matmul(p1, gsp_1, proj_shape, proj_shape)
        del p1, gsp_1

        coeff = self.physics.smearing(
            tr_vectors,
            eigenvecs,
            eigenvals,
            smearing_temperature,
            mu,
            nsub,
            self.n_orb,
            is_dual=False,
            spin=None,
        )

        gsp_2 = self.physics.get_proj(coeff, tr_vectors, nsub, self.n_orb, is_dual=False)
        del coeff, tr_vectors
        loc_z2_op_2 = self.backend.matmul(p2, gsp_2, proj_shape, proj_shape)
        del p2, gsp_2

        # get the diagonal elements
        chern_diags_1 = self.backend.get_diag(loc_z2_op_1, N=self.n_orb)
        del loc_z2_op_1

        chern_diags_2 = self.backend.get_diag(loc_z2_op_2, N=self.n_orb)
        del loc_z2_op_2

        chern_diags_1 = (
            (
                -np.imag(chern_diags_1)
                * (self.Lx * self.Ly)
                / (np.pi * (8 if formula == "symmetric" else 2))
            )
            if self.backend.is_master_rank
            else None
        )

        chern_diags_2 = (
            (
                -np.imag(chern_diags_2)
                * (self.Lx * self.Ly)
                / (np.pi * (8 if formula == "symmetric" else 2))
            )
            if self.backend.is_master_rank
            else None
        )

        chern_diags_1 = self.backend.shared_array(chern_diags_1)
        chern_diags_2 = self.backend.shared_array(chern_diags_2)

        if bare_marker:
            return np.array([chern_diags_1, chern_diags_2])

        # If macroscopic_average I have to compute the lattice values with the averages
        #   first (explicit PBC contraction passed)
        if macroscopic_average:
            contraction = pbc_lattice_contraction(self, cutoff)
            chern_diags_1 = average_over_radius(
                self, chern_diags_1, cutoff, contraction=contraction
            )
            chern_diags_2 = average_over_radius(
                self, chern_diags_2, cutoff, contraction=contraction
            )

        if direction is not None:
            pbclcm_1 = self._extract_line_marker(
                chern_diags_1, direction, start, macroscopic_average
            )
            pbclcm_2 = self._extract_line_marker(
                chern_diags_2, direction, start, macroscopic_average
            )

            if not return_projector:
                return np.abs(np.fmod(0.5 * (pbclcm_1 - pbclcm_2), 2))
            else:
                return np.abs(np.fmod(0.5 * (pbclcm_1 - pbclcm_2), 2)), np.asarray(
                    return_proj
                )

        # If not macroscopic averages I sum the values of the Chern operators of the unit cell
        if not macroscopic_average:
            chern_diags_1 = self._trace_unit_cell(chern_diags_1)
            chern_diags_2 = self._trace_unit_cell(chern_diags_2)

        if not return_projector:
            return np.abs(np.fmod(0.5 * (chern_diags_1 - chern_diags_2), 2))
        else:
            return np.abs(np.fmod(0.5 * (chern_diags_1 - chern_diags_2), 2)), np.array(
                return_proj
            )
