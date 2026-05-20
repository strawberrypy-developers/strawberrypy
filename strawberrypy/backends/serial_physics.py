import numpy as np
from opt_einsum import contract
import scipy.linalg as la


class SerialPhysics:
    def __init__(self, **kwargs):
        super().__init__()

    def fermidirac(self, evals, temperature: float, mu: float, **kwargs):
        r"""
        The Fermi-Dirac distribution :math:`f(\epsilon, T, \mu) = \big[ 1 +
        e^{\frac{\epsilon-\mu}{T}} \big]^{-1}`.

        Parameters
        ----------
            evals :
                List of eigenvalues of the Hamiltonian.
            temperature :
                Temperature of the system.
            mu :
                The chemical potential of the system.

        Returns
        -------
            occupations : :python:`np.array | float`
                The occupation(s) of the state corresponding to the given energy(ies).
        """
        if temperature < 1e-6:
            if evals.shape == ():
                return 1 if evals < +mu else 0
            else:
                return np.where(evals <= mu, 1.0, 0.0)
        else:
            return 1 / (1 + np.exp((evals - mu) / temperature))

    def chemical_potential(
        self, evals, temperature: float, occupied_states: int, **kwargs
    ):
        r"""Calculate the chemical potential of a given model. This is done by
        knowing the eigenvalue distribution and the number of electrons (occupied states)
        in the system.

        Parameters
        ----------
            evals :
                List of eigenvalues of the Hamiltonian.
            temperature :
                Temperature (real or fictitious, as in the case of smearing) of the system,
                appearing in the Fermi-Dirac distribution.
            occupied_states :
                Number of occupied states of the system.
            kwargs :
                Additional keyword arguments, such as:

                - `maxiter` : Maximum number of iterations for the bisection method to find
                    the chemical potential. Default is 200.

        Returns
        -------
            mu : :python:`float`
                The chemical potential of the system.
        """
        mu_min = np.min(evals)
        mu_max = np.max(evals)
        mu = 0
        niter = 0
        maxiter = kwargs.get("maxiter", 300)

        while True:
            mu = 0.5 * (mu_min + mu_max)
            n_exp = np.sum(self.fermidirac(evals, temperature, mu))

            if n_exp < occupied_states:
                mu_min = mu
            else:
                mu_max = mu

            if np.abs(n_exp - occupied_states) < 1e-6:
                break

            niter += 1
            if niter > maxiter:
                raise RuntimeError(
                    f"Chemical potential cannot be found: bisection method failed ({maxiter} iterations)"
                )
        return mu

    def _add_occupied_states(self, evals, fd_cut, nocc_t0, ntba, **kwargs):
        r"""Functionality to determine the smearing temperature and the chemical
        potential such that the number of occupied states is nocc_t0 + ntba.

        Parameters
        ----------
            evals :
                Eigenvales of the Hamiltonian at the :math:`\Gamma`-point.
            fd_cut :
                Cutoff imposed on the Fermi-Dirac distribution.
            nocc_t0 :
                Number of occupied states at zero temperature.
            ntba :
                Number of states to be added relative to the a priori set filling at zero
                temperature.
            kwargs :
                Additional keyword arguments, such as:

                - `T_min` : Smallest temperature for the bisection procedure. Default is 0.
                - `T_max` : Highest temperature for the bisection procedure. Default is 0.1.
                - `max_iter` : Maximum number of iterations for the bisection method to find
                    the chemical potential. Default is 200.

        Returns
        -------
            nocc :
                Number of occupied states after adding at finite temperature.
            temperature :
                Finite temperature for which the number of occupied states is nocc_t0 + ntba.
            mu :
                Chemical potential.
        """
        tmin = kwargs.get("T_min", 0.0)
        tmax = kwargs.get("T_max", 0.1)
        maxiter = kwargs.get("max_iter", 200)

        nocc = 0
        temperature = 0
        mu = 0

        niter = 0
        if ntba == 0:
            return nocc_t0, 0.0, self.chemical_potential(evals, 0.0, nocc_t0)
        else:
            while True:
                temperature = 0.5 * (tmin + tmax)
                mu = self.chemical_potential(evals, temperature, nocc_t0)
                nocc = np.sum(self.fermidirac(evals, temperature, mu) > fd_cut)
                if nocc < nocc_t0 + ntba:
                    tmin = temperature
                else:
                    tmax = temperature
                if np.abs(nocc - nocc_t0 - ntba) < 1e-6:
                    break
                niter += 1
                if niter > maxiter:
                    print(
                        f"Max iterations reached: leaving default values ({maxiter} iterations)"
                    )
                    return nocc_t0, 0.0, self.chemical_potential(evals, 0.0, nocc_t0)
            return nocc, temperature, mu

    def _get_occupied_states(self, evals, temperature, mu, fd_cut, **kwargs):
        r"""Functionality to determine the number of occupied states given a
        smearing temperature and chemical potential.

        Parameters
        ----------
            evals :
                Eigenvales of the Hamiltonian at the :math:`\Gamma`-point.
            temperature :
                Fictitious temperature :math:`T_s` used for smearing.
            mu :
                Chemical potential.
            fd_cut :
                Cutoff for the Fermi-Dirac distribution.

        Returns
        -------
            nocc :
                Number of occupied states at the given temperature and chemical potential.
        """
        return np.sum(self.fermidirac(evals, temperature, mu) > fd_cut)

    def _add_states_until_gap(self, evals, nocc_t0, gap_tol, fd_cut, **kwargs):
        r"""Functionality to determinate the smallest number of states to be added
        to nocc_t0 such that the system has a gap larger than gap_tol.

        Parameters
        ----------
            evals :
                Eigenvales of the Hamiltonian at the :math:`\Gamma`-point.
            nocc_t0 :
                Number of occupied states at zero temperature.
            gap_tol :
                Minimum value of the gap when adding states to :python:`nocc_t0`. Default
                is :python:`1e-8`.
            fd_cut :
                Cutoff imposed on the Fermi-Dirac distribution.

            kwargs :
                Additional keyword arguments, such as:
                - `T_min` : Smallest temperature for the bisection procedure. Default is 0.
                - `T_max` : Highest temperature for the bisection procedure. Default is 0.1.
                - `max_iter` : Maximum number of iterations for the bisection method to
                    find the chemical potential. Default is 200.

        Returns
        -------
            nocc :
                Minimum number of states before finding a gap.
            t_f :
                Minimum finite temperature before finding a gap.
            mu_f :
                Chemical potential.
        """
        n_tba = 0
        for i in range(nocc_t0):
            gap = evals[nocc_t0 + i] - evals[nocc_t0 + i - 1]
            if gap < gap_tol:
                continue
            else:
                n_tba = i
                break

        nocc, temperature, mu = self._add_occupied_states(
            evals, fd_cut, nocc_t0, ntba=n_tba, **kwargs
        )
        temp = np.linspace(0.0, temperature, 101)
        t_f = temperature
        mu_f = mu
        for t in reversed(temp):
            mu_t = self.chemical_potential(evals, t, nocc_t0)
            nocc_t = np.sum(self.fermidirac(evals, t, mu_t) > fd_cut)
            if nocc_t < nocc:
                break
            else:
                t_f = t
                mu_f = mu_t

        print(
            "ADD OCC STATES:\nn_states={0}\ntemperature_min={1}\nmu={2}\ngap={3}".format(
                nocc, t_f, mu_f, gap
            )
        )
        return nocc, t_f, mu_f

    def smearing(
        self,
        vecs,
        gamma_hevecs,
        evals,
        temperature: float,
        mu: float,
        n_states,
        n_orb,
        is_dual=False,
        spin=None,
        **kwargs,
    ):
        r"""Smearing coefficients for a given set of states.

        Smearing introduced to improve the convergence of the formula: it
        measures how much the projector built from :python:`vecs` is similar to
        the one built from :python:`gamma_hevecs`, the eigenstates of the
        Hamiltonian at the :math:`\Gamma`-point. Naming :math:`|\phi_n\rangle`
        the vectors in :python:`vecs` and :math:`|u_n\rangle` the ones in
        :python:`gamma_hevecs`, the smearing factor is computed as :math:`c_n =
        \sum_m f(\epsilon_m, T_s, \mu)|\langle \phi_n|u_m\rangle|^2`, where
        :math:`\epsilon_m` is the eigenvalue corresponding to the eigenstate
        :math:`|u_m\rangle`, :math:`T_s` is the smearing temperature,
        :math:`\mu` is the chemical potential.

        Parameters
        ----------
            vecs :
                States that need to be weighted according to some smearing.
            gamma_hevecs :
                Eigenstates of th Hamiltonian at the :math:`\Gamma`-point.
            evals :
                Eigenvales of the Hamiltonian at the :math:`\Gamma`-point.
            temperature :
                Temperature introduced to smoothen the occupation of the states (smearing
                temperature).
            mu :
                Chemical potential of the system.
            n_orb :
                Number of orbitals in the system.
            n_states :
                Number of states to take the weights.
            is_dual :
                Whether the states in `vecs` are in dual (bra) space.
            spin :
                Spin index for the states. If None, all spins are considered. Can be 'up' or
                'down' if the system is spin-polarized.
        Returns
        -------
            smearing_coeffs : :python:`np.array`
                A list of smearing coefficients that weights the states :python:`vecs`.
        """
        n_sub = n_states // 2

        if spin is None:
            vecs_len = n_states
            vecs = vecs[:n_states, :n_orb].T if is_dual else vecs[:n_orb, :n_states]
        elif spin == "down":
            vecs_len = n_sub
            vecs = vecs[:n_sub, :n_orb].T if is_dual else vecs[:n_orb, :n_sub]
        else:
            vecs_len = n_sub
            vecs = vecs[:n_sub, :n_orb].T if is_dual else vecs[:n_orb, n_sub:]

        if temperature < 1e-6:
            return np.ones(vecs_len)
        else:
            # Overlap matrix: shape (n, m)
            # n: vecs.shape[1]=n_states, m: gamma_hevecs.shape[1]=n_orb
            overlaps = np.abs(vecs.conj().T @ gamma_hevecs) ** 2

            # Fermi-Dirac weights: shape (m=n_orb,)
            weights = self.fermidirac(evals, temperature, mu)

            # Weighted sum over m
            return overlaps @ weights  # shape: (n_states,)

    def get_proj(
        self, coeff, vecs, n_states, n_orb=None, spin=None, is_dual=False, *args, **kwargs
    ):
        r"""Get the projector over a given subspace weighted by some coefficients.

        Get the projector on the states in :python:`vecs` weighted by the smearing
        coefficients in :python:`coeff`. The projector is computed as :math:`P = \sum_n c_n
        |\phi_n\rangle\langle \phi_n|`, where :math:`c_n` are the smearing coefficients and
        :math:`|\phi_n\rangle` are the states in :python:`vecs`.

        Parameters
        ----------
            coeff :
                Smearing coefficients for the states in :python:`vecs`.
            vecs :
                States on which the projector is built.
            n_states :
                Number of states to take into account for the projector.
            n_orb :
                Number of orbitals in the system.
            spin :
                The spin sector for which the projector is computed. Default is :python:`None`,
                referring to a spinless case.
            is_dual :
                Whether the states in :python:`vecs` are dual states. Default is :python:`False`.

        Returns
        -------
            P : :python:`np.array`
                The projector on the states in :python:`vecs` weighted by the smearing
                coefficients in :python:`coeff`.
        """
        if coeff.ndim == 2:
            coeff = np.diag(coeff)

        n_sub = n_states // 2

        if spin == None:
            vecs = vecs[:n_states, :n_orb].T if is_dual else vecs[:n_orb, :n_states]
        elif spin == "down":
            vecs = vecs[:n_sub, :n_orb].T if is_dual else vecs[:n_orb, :n_sub]
        else:
            vecs = vecs[:n_sub, :n_orb].T if is_dual else vecs[:n_orb, n_sub:]

        P = contract("ji,ki->jk", vecs * coeff.reshape(1, -1), vecs.conj())

        return P

    #################################################
    # Functions used in the Z2 topological markers
    #################################################
    def _delta_projection(
        self, evecs, rank: int, trial_projections=None, states_uc=None, n_orb=None
    ):
        r"""Compute the quasi Wannier functions via projections.

        Parameters
        ----------
            evecs :
                The eigenvectors of the Hamiltonian.
            rank :
                The number of effectively occupied states (in the case of
                :python:`smearing_temperature == 0`, half-filling is assumed).
            trial_projections :
                Matrix of the trial projections used in the procedure.

        Returns
        -------
            qwfs :
                The list of quasi Wannier functions by row.

        .. note::
            The parameter :python:`trial_projections` should contain the projections
            **in the primitive cell**, as the rest are computed replicating of this choice.
        """

        # If no trial projection is specified, use a default one if the number of atoms is 2, else return error
        projections = trial_projections
        if projections is None and states_uc // 2 == 2:
            projections = np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, -1, 0]]
            )
        elif projections is None:
            raise RuntimeError(
                "Initial projections are not specified and cannot be guessed."
            )

        # Normalize the matrix of projections
        one = np.kron(np.eye(rank // 2), projections) / la.norm(projections[:, 0])

        # Compute rotation
        amatrix = evecs[:, :rank].conjugate().T @ one[:, ::2][:, :rank]
        smatrix = amatrix.T.conjugate() @ amatrix
        smatrix = np.array(la.sqrtm(la.pinv(smatrix)), dtype=complex)

        # Compute quasi Wannier functions
        qwfs = contract("ki,ij->kj", evecs[:, :rank], amatrix @ smatrix)
        qwfs = qwfs / np.linalg.norm(qwfs, axis=0, keepdims=True)
        return qwfs  # shape: (n_orb, rank)

    def _time_reversal_separation(self, evecs_proj, *args, **kwargs):
        r"""Split the quasi Wannier functions using the time reversal symmetry.

        Parameters
        ----------
            evecs_proj :
                The quasi Wannier functions, ordered by row.

        Returns
        -------
            vectors, tr_evecs :
                The quasi Wannier functions split using time reversal.
        """
        # Sigma y matrix
        pauli_y = np.array([[0, -1j], [1j, 0]])
        sigma_y = np.kron(np.eye(evecs_proj.shape[0] // 2), pauli_y)

        # I choose one eigenvector and compute its time reversal partner
        tr_evecs = 1.0j * sigma_y @ evecs_proj[:, ::2].conj()
        vectors = evecs_proj[:, ::2]

        return vectors, tr_evecs  # shape: (n_orb, rank//2)
