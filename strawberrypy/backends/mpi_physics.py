import numpy as np
from mpi4py import MPI
import scipy.linalg as la

from .mpi_linalg import MPILinalg


class MPIPhysics(MPILinalg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fermidirac(
        self, evals, temperature: float, mu: float, root: int = 0, broadcast: bool = False
    ):
        r"""The Fermi-Dirac distribution :math:`f(\epsilon, T, \mu)=\big[ 1 +
        e^{\frac{\epsilon-\mu}{T}} \big]^{-1}`.

        Parameters
        ----------
            evals :
                List of eigenvalues of the Hamiltonian. This should be provided on `root`
                rank. On other ranks, it can be None or any value since it will be ignored.
            temperature :
                Temperature of the system.
            mu :
                The chemical potential of the system.
            root :
                The rank that provides the input `evals` and receives the full occupations.
                Default is 0.
            broadcast :
                If True, the full occupations will be broadcast to all ranks after being
                computed on `root`.

        Returns
        -------
            occupations : :python:`np.array | float`
                The occupation(s) of the state corresponding to the given energy(ies)
        """
        comm = self.comm
        rank = self.mpi_rank
        size = self.mpi_size

        # Determine if we are dealing with a scalar evals
        is_scalar = False
        if rank == root:
            if np.isscalar(evals) or getattr(evals, "shape", ()) == ():
                is_scalar = True
        is_scalar = comm.bcast(is_scalar, root=root)

        # Handle scalar evals: compute on root, optionally broadcast
        if is_scalar:
            if rank == root:
                if temperature < 1e-6:
                    occ_local = 1.0 if evals <= mu else 0.0
                else:
                    occ_local = 1.0 / (1.0 + np.exp((float(evals) - mu) / temperature))
            else:
                occ_local = None

            if broadcast:
                occ = comm.bcast(occ_local, root=root)
                return occ
            else:
                return occ_local if rank == root else None

        # At this point we expect `evals` to be an array on `root`
        if rank == root:
            if evals is None:
                raise ValueError("Evals must be provided on root rank")
            evals = np.ascontiguousarray(np.asarray(evals, dtype=np.float64))
            N = evals.size
            base = N // size
            extras = N % size
            counts = [base + (1 if i < extras else 0) for i in range(size)]
            displs = [sum(counts[:i]) for i in range(size)]
        else:
            N = None
            counts = None
            displs = None

        # Broadcast metadata
        N = comm.bcast(N, root=root)
        counts = comm.bcast(counts, root=root)
        displs = comm.bcast(displs, root=root)

        # Prepare receive buffer for this rank
        recv_count = counts[rank]
        recvbuf = np.empty(recv_count, dtype=np.float64)

        # Scatter chunks of `evals` from root to all ranks
        if rank == root:
            comm.Scatterv([evals, counts, displs, MPI.DOUBLE], recvbuf, root=root)
        else:
            comm.Scatterv([None, counts, displs, MPI.DOUBLE], recvbuf, root=root)

        # Compute local occupations
        if recv_count > 0:
            if temperature < 1e-6:
                local_occ = np.where(recvbuf <= mu, 1.0, 0.0)
            else:
                local_occ = 1.0 / (1.0 + np.exp((recvbuf - mu) / temperature))
        else:
            local_occ = np.empty(0, dtype=np.float64)

        # Gather results back to root
        if rank == root:
            global_occ = np.empty(N, dtype=np.float64)
            comm.Gatherv(local_occ, [global_occ, counts, displs, MPI.DOUBLE], root=root)
        else:
            comm.Gatherv(local_occ, [None, counts, displs, MPI.DOUBLE], root=root)

        if broadcast:
            # Collective: broadcast the full occupations to all ranks
            if rank == root:
                to_bcast = global_occ
            else:
                to_bcast = None
            occs = comm.bcast(to_bcast, root=root)
            return occs
        else:
            return global_occ if rank == root else None

    def chemical_potential(
        self,
        evals,
        temperature: float,
        occupied_states: int,
        root: int = 0,
        broadcast: bool = True,
        **kwargs,
    ):
        r"""Calculate the chemical potential of a given model. This is done by
        knowing the eigenvalue distribution and the number of electrons (occupied states)
        in the system.

        Parameters
        ----------
            evals :
                List of eigenvalues of the Hamiltonian. This should be provided on `root`
                rank. On other ranks, it can be None or any value since it will be ignored.
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
        comm = self.comm
        rank = self.mpi_rank
        size = self.mpi_size

        # Expect `evals` to live on `root` rank
        if rank == root:
            if evals is None:
                raise ValueError("Evals must be provided on root rank")
            evals = np.ascontiguousarray(np.asarray(evals, dtype=np.float64))
            N = evals.size
            base = N // size
            extras = N % size
            counts = [base + (1 if i < extras else 0) for i in range(size)]
            displs = [sum(counts[:i]) for i in range(size)]
            mu_min = np.min(evals)
            mu_max = np.max(evals)
        else:
            N = None
            counts = None
            displs = None
            mu_min = None
            mu_max = None

        maxiter = kwargs.get("maxiter", 200)

        # Broadcast metadata and mu bounds
        N = comm.bcast(N, root=root)
        counts = comm.bcast(counts, root=root)
        displs = comm.bcast(displs, root=root)
        mu_min = comm.bcast(mu_min, root=root)
        mu_max = comm.bcast(mu_max, root=root)

        # Scatter eigenvalues once outside the loop
        recv_count = counts[rank]
        recvbuf = np.empty(recv_count, dtype=np.float64)
        if rank == root:
            comm.Scatterv([evals, counts, displs, MPI.DOUBLE], recvbuf, root=root)
        else:
            comm.Scatterv([None, counts, displs, MPI.DOUBLE], recvbuf, root=root)

        mu = 0.0
        converged = False

        for niter in range(maxiter + 1):
            if rank == root:
                mu = 0.5 * (mu_min + mu_max)
            # Broadcast current mu to all ranks
            mu = comm.bcast(mu, root=root)

            # Compute local expected number of states
            if recvbuf.size == 0:
                local_nexp = 0.0
            else:
                if temperature < 1e-6:
                    local_nexp = float(np.sum(recvbuf <= mu))
                else:
                    local_nexp = float(
                        np.sum(1.0 / (1.0 + np.exp((recvbuf - mu) / temperature)))
                    )

            # Reduce to root
            global_nexp = comm.reduce(local_nexp, op=MPI.SUM, root=root)

            if rank == root:
                if global_nexp < occupied_states:
                    mu_min = mu
                else:
                    mu_max = mu

                if np.abs(global_nexp - occupied_states) < 1e-6:
                    converged = True

            # Broadcast convergence flag and updated bounds for next iteration
            converged = comm.bcast(converged, root=root)
            mu_min = comm.bcast(mu_min, root=root)
            mu_max = comm.bcast(mu_max, root=root)

            if converged:
                break

        if not converged and rank == root:
            raise RuntimeError(
                f"Chemical potential cannot be found: bisection method failed ({maxiter} iterations)"
            )

        if broadcast:
            mu = comm.bcast(mu, root=root)
            return mu
        else:
            return mu if rank == root else None

    def _add_occupied_states(
        self,
        evals,
        fd_cut,
        nocc_t0,
        ntba,
        root: int = 0,
        broadcast: bool = True,
        **kwargs,
    ):
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
        comm = self.comm
        rank = self.mpi_rank
        size = self.mpi_size

        tmin = kwargs.get("T_min", 0.0)
        tmax = kwargs.get("T_max", 0.1)
        maxiter = kwargs.get("max_iter", 200)

        # Fast path: no added states
        if ntba == 0:
            mu = self.chemical_potential(
                evals,
                0.0,
                nocc_t0,
                root=root,
                broadcast=broadcast,
                maxiter=kwargs.get("maxiter", 200),
            )
            if broadcast:
                return nocc_t0, 0.0, mu
            else:
                return (nocc_t0, 0.0, mu) if rank == root else (None, None, None)

        # Expect evals on root; prepare scatter metadata
        if rank == root:
            if evals is None:
                raise ValueError("Evals must be provided on root rank")
            evals = np.ascontiguousarray(np.asarray(evals, dtype=np.float64))
            N = evals.size
            base = N // size
            extras = N % size
            counts = [base + (1 if i < extras else 0) for i in range(size)]
            displs = [sum(counts[:i]) for i in range(size)]
        else:
            N = None
            counts = None
            displs = None

        N = comm.bcast(N, root=root)
        counts = comm.bcast(counts, root=root)
        displs = comm.bcast(displs, root=root)

        # Scatter eigenvalues once for local counting
        recv_count = counts[rank]
        recvbuf = np.empty(recv_count, dtype=np.float64)
        if rank == root:
            comm.Scatterv([evals, counts, displs, MPI.DOUBLE], recvbuf, root=root)
        else:
            comm.Scatterv([None, counts, displs, MPI.DOUBLE], recvbuf, root=root)

        nocc = 0
        temperature = 0.0
        mu = 0.0

        converged = False
        for _ in range(maxiter + 1):
            temperature = 0.5 * (tmin + tmax)

            # Compute chemical potential (broadcast to all ranks so they can count)
            mu = self.chemical_potential(
                evals,
                temperature,
                nocc_t0,
                root=root,
                broadcast=True,
                maxiter=kwargs.get("maxiter", 200),
            )

            # Compute local number of states above fd_cut
            if recvbuf.size == 0:
                local_nocc = 0.0
            else:
                if temperature < 1e-6:
                    local_nocc = float(np.sum(recvbuf <= mu))
                else:
                    occs = 1.0 / (1.0 + np.exp((recvbuf - mu) / temperature))
                    local_nocc = float(np.sum(occs > fd_cut))

            # Reduce to root
            global_nocc = comm.reduce(local_nocc, op=MPI.SUM, root=root)

            if rank == root:
                nocc = int(global_nocc)
                if nocc < nocc_t0 + ntba:
                    tmin = temperature
                else:
                    tmax = temperature

                if np.abs(nocc - nocc_t0 - ntba) < 1e-6:
                    converged = True

            # Broadcast convergence and updated bounds
            converged = comm.bcast(converged, root=root)
            tmin = comm.bcast(tmin, root=root)
            tmax = comm.bcast(tmax, root=root)

            if converged:
                break

        if not converged:
            mu = self.chemical_potential(
                evals, 0.0, nocc_t0, root=root, broadcast=broadcast
            )
            if rank == root:
                print(
                    f"Max iterations reached: leaving default values ({maxiter} iterations)"
                )
            if broadcast:
                return nocc_t0, 0.0, mu
            else:
                return (nocc_t0, 0.0, mu) if rank == root else (None, None, None)

        # Finalize return values
        if broadcast:
            # Ensure all ranks have mu and nocc and temperature
            mu = comm.bcast(mu, root=root)
            nocc = comm.bcast(nocc if rank == root else None, root=root)
            temperature = comm.bcast(temperature if rank == root else None, root=root)
            return nocc, temperature, mu
        else:
            return (nocc, temperature, mu) if rank == root else (None, None, None)

    def _get_occupied_states(
        self, evals, temperature, mu, fd_cut, root: int = 0, broadcast: bool = True
    ):
        r"""Functionality to determine the number of occupied states given a
        smearing temperature and chemical potential.

        Parameters
        ----------
            evals :
                Eigenvales of the Hamiltonian at the :math:`\Gamma`-point. This should be
                provided on `root` rank. On other ranks, it can be None or any value since
                it will be ignored.
            temperature :
                Fictitious temperature :math:`T_s` used for smearing.
            mu :
                Chemical potential.
            fd_cut :
                Cutoff for the Fermi-Dirac distribution.
            root :
                The rank that provides the input `evals` and receives the total count.
                Default is 0.
            broadcast :
                If True, the count will be broadcast to all ranks.

        Returns
        -------
            nocc :
                Number of occupied states at the given temperature and chemical potential.
        """
        comm = self.comm
        rank = self.mpi_rank
        size = self.mpi_size

        # Expect evals on root
        if rank == root:
            if evals is None:
                raise ValueError("evals must be provided on root rank")
            evals = np.ascontiguousarray(np.asarray(evals, dtype=np.float64))
            N = evals.size
            base = N // size
            extras = N % size
            counts = [base + (1 if i < extras else 0) for i in range(size)]
            displs = [sum(counts[:i]) for i in range(size)]
        else:
            N = None
            counts = None
            displs = None

        N = comm.bcast(N, root=root)
        counts = comm.bcast(counts, root=root)
        displs = comm.bcast(displs, root=root)

        # Scatter eigenvalues
        recv_count = counts[rank]
        recvbuf = np.empty(recv_count, dtype=np.float64)
        if rank == root:
            comm.Scatterv([evals, counts, displs, MPI.DOUBLE], recvbuf, root=root)
        else:
            comm.Scatterv([None, counts, displs, MPI.DOUBLE], recvbuf, root=root)

        # Compute local occupations
        if recvbuf.size == 0:
            local_occs = np.empty(0, dtype=np.float64)
        else:
            if temperature < 1e-6:
                local_occs = np.where(recvbuf <= mu, 1.0, 0.0)
            else:
                local_occs = 1.0 / (1.0 + np.exp((recvbuf - mu) / temperature))

        # Count local states above fd_cut
        local_nocc = float(np.sum(local_occs > fd_cut))

        # Reduce to root
        global_nocc = comm.reduce(local_nocc, op=MPI.SUM, root=root)

        if broadcast:
            global_nocc = comm.bcast(global_nocc if rank == root else None, root=root)
            return int(global_nocc)
        else:
            return int(global_nocc) if rank == root else None

    def _add_states_until_gap(self, evals, nocc_t0, gap_tol, fd_cut, root=0, **kwargs):
        r"""Functionality to determinate the smallest number of states to be added
        to nocc_t0 such that the system has a gap larger than gap_tol.

        Parameters
        ----------
            evals :
                Eigenvales of the Hamiltonian at the :math:`\Gamma`-point. This should be
                provided on `root` rank. On other ranks, it can be None or any value since it will be ignored.
            nocc_t0 :
                Number of occupied states at zero temperature.
            gap_tol :
                Minimum value of the gap when adding states to :python:`nocc_t0`. Default is :python:`1e-8`.
            fd_cut :
                Cutoff imposed on the Fermi-Dirac distribution.
            root :
                The rank that provides the input `evals` and receives the total count.
                Default is 0.
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
        comm = self.comm
        rank = self.mpi_rank

        n_tba = 0
        gap = 0.0

        if rank == root:
            for i in range(nocc_t0):
                gap = evals[nocc_t0 + i] - evals[nocc_t0 + i - 1]
                if gap < gap_tol:
                    continue
                else:
                    n_tba = i
                    break

        n_tba = comm.bcast(n_tba, root=root)
        gap = comm.bcast(gap, root=root)

        nocc, temperature, mu = self._add_occupied_states(
            evals, fd_cut, nocc_t0, ntba=n_tba, root=root, **kwargs
        )
        temp = np.linspace(0.0, temperature, 101)
        t_f = temperature
        mu_f = mu
        for t in reversed(temp):
            mu_t = self.chemical_potential(evals, t, nocc_t0, root=root)
            nocc_t = self._get_occupied_states(evals, t, mu_t, fd_cut, root=root)
            if nocc_t < nocc:
                break
            else:
                t_f = t
                mu_f = mu_t

        if rank == root:
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
    ):
        r"""Smearing coefficients for a given set of states.

        Smearing introduced to improve the convergence of the formula: it
        measures how much the projector built from :python:`vecs` is similar to
        the one built from :python:`gamma_hevecs`, the eigenstates of the
        Hamiltonian at the :math:`\Gamma`-point. Naming :math:`|\phi_n\rangle`
        the vectors in :python:`vecs` and :math:`|u_n\rangle` the ones in
        :python:`gamma_hevecs`, the smearing factor is computed as
        :math:`c_n=\sum_m f(\epsilon_m, T_s, \mu)|\langle
        \phi_n|u_m\rangle|^2`, where :math:`\epsilon_m` is the eigenvalue
        corresponding to the eigenstate :math:`|u_m\rangle`, :math:`T_s` is the
        smearing temperature, :math:`\mu` is the chemical potential.

        Parameters
        ----------
            vecs :
                List of states that need to be weighted according to some smearing.
            gamma_hevecs :
                Eigenstates of th Hamiltonian at the :math:`\Gamma`-point.
            evals :
                Eigenvales of the Hamiltonian at the :math:`\Gamma`-point. This should be
                provided on `root` rank. On other ranks, it can be None or any value since
                it will be ignored.
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
                Whether the states in `vecs` are in dual (bra) space. The matrix has different
                shape and the matmul call needs to be adapted accordingly.
            spin :
                Spin index for the states. If None, all spins are considered. Can be 'up' or
                'down' if the system is spin-polarized.

        Returns
        -------
            smearing_coeffs : :python:`np.array`
                A list of smearing coefficients that weights the states :python:`vecs`.
        """
        n_sub = n_states // 2

        if spin == None:
            slc_idx_vecs = [[0, n_orb], [0, n_states]]
            vecs_len = n_states
        elif spin == "down":
            slc_idx_vecs = [[0, n_orb], [0, n_sub]]
            vecs_len = n_sub
        else:
            slc_idx_vecs = [[0, n_orb], [n_sub, n_sub * 2]]
            vecs_len = n_sub

        if temperature < 1e-6:
            if self.mpi_rank == 0:
                weights = np.ones(vecs_len)
            else:
                weights = None

            weights = self.distribute_diag(weights, root=0)
            return weights

        else:
            slc_idx_hevecs = [[0, n_orb], [0, n_orb]]
            glob_shape_hevecs = (n_orb, n_orb)
            glob_shape_vecs = [n_orb, n_states]

            if is_dual:
                slc_idx_vecs.reverse()
                glob_shape_vecs.reverse()

            overlap = self.matmul(
                vecs,
                gamma_hevecs.conj(),
                glob_shape_vecs,
                glob_shape_hevecs,
                "N" if is_dual else "T",
                "N",
                slc_idx_vecs,
                slc_idx_hevecs,
            )

            fd = self.fermidirac(evals, temperature, mu, root=0, broadcast=False)
            fd = self.distribute_diag(fd, root=0)

            tmp = self.matmul(overlap, fd, (vecs_len, n_orb), (n_orb, n_orb))

            coeffs = self.matmul(
                tmp, overlap.conj(), (vecs_len, n_orb), (vecs_len, n_orb), "N", "T"
            )

            #  We only need the diagonal elements, which correspond to the smearing
            #   coefficients for each state with itself. However, due to the distributed
            #   nature of the computation, we may have non-zero off-diagonal elements that
            #   correspond to overlaps between different states. To ensure that we only
            #   return the diagonal smearing coefficients, we set all off-diagonal
            #   elements to zero
            coeffs = self.set_zero_off_diag(coeffs, vecs_len)

            return coeffs

    def get_proj(self, coeff, vecs, n_states, n_orb, spin=None, is_dual=False):
        r"""Get the projector over a given subspace weighted by some coefficients.

        Get the projector on the states in :python:`vecs` weighted by the
        smearing coefficients in :python:`coeff`. The projector is computed as
        :math:`P=\sum_n c_n |\phi_n\rangle\langle \phi_n|`, where :math:`c_n`
        are the smearing coefficients and :math:`|\phi_n\rangle` are the states
        in :python:`vecs`.

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
            is_ground_state :
                Whether the states in :python:`vecs` are the ground state ones.

        Returns
        -------
            P : :python:`np.array`
                The projector on the states in :python:`vecs` weighted by the smearing
                coefficients in :python:`coeff`.
        """
        n_sub = n_states // 2
        if spin == "down":
            slc_idx_states = [[0, n_orb], [0, n_sub]]
            glob_shape_vecs = [n_orb, n_states]
        elif spin == "up":
            slc_idx_states = [[0, n_orb], [n_sub, n_sub * 2]]
            glob_shape_vecs = [n_orb, n_states]
        else:
            slc_idx_states = [[0, n_orb], [0, n_states]]
            glob_shape_vecs = [n_orb, n_states]
            n_sub = n_states

        if is_dual:
            slc_idx_states.reverse()
            glob_shape_vecs.reverse()

        tmp = self.matmul(
            vecs,
            coeff,
            glob_shape_vecs,
            (n_sub, n_sub),
            "T" if is_dual else "N",
            "N",
            slc_idx_A=slc_idx_states,
        )

        P = self.matmul(
            tmp,
            vecs.conj(),
            (n_orb, n_sub),
            glob_shape_vecs,
            "N",
            "N" if is_dual else "T",
            slc_idx_B=slc_idx_states,
        )

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
        # If no trial projection is specified, use a default one if the number of atoms
        #   is 2, else return error
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
        one = (
            np.kron(np.eye(rank // 2), projections) / la.norm(projections[:, 0])
            if self.is_master_rank
            else None
        )
        one = self.distribute(one[:, ::2] if self.is_master_rank else None)

        # Compute rotation
        amatrix = self.matmul(
            evecs.conj(),
            one,
            (n_orb, n_orb),
            (rank // 2 * 4, rank // 2 * 2),
            "T",
            "N",
            [[0, n_orb], [0, rank]],
            [[0, n_orb], [0, rank]],
        )

        # Assume that smatrix is symmetric positive definite
        smatrix = self.matmul(
            amatrix.conj(), amatrix, (rank, rank), (rank, rank), "T", "N"
        )

        eigvals, eigvecs = self.eigh(
            smatrix, rank, rank, collect_evec=False, debug_info=False
        )

        # Take the inverse square root of the eigenvalues
        eigvals_sqrt = self.distribute(np.diag(eigvals ** (-0.5)))
        tmp = self.matmul(eigvecs, eigvals_sqrt, (rank, rank), (rank, rank), "N", "N")
        smatrix = self.matmul(
            tmp, eigvecs.conj(), (rank, rank), (rank, rank), "N", "T"
        )  # This is the inverse square root of smatrix

        tmp = self.matmul(amatrix, smatrix, (rank, rank), (rank, rank), "N", "N")

        qwfs = self.matmul(
            evecs,
            tmp,
            (n_orb, n_orb),
            (rank, rank),
            "N",
            "N",
            slc_idx_A=[[0, n_orb], [0, rank]],
        )  # Dimension of [n_orb,rank]

        # Compute quasi Wannier functions
        # Normalization of the quasi Wannier functions
        norm = self.matmul(qwfs.conj(), qwfs, (n_orb, rank), (n_orb, rank), "T", "N")
        norm = self.get_diag(norm, N=rank) ** 0.5

        norm_matrix = None
        if self.is_master_rank:
            norm_matrix = np.tile(norm, (n_orb, 1))

        norm_matrix = self.distribute(norm_matrix)

        return qwfs / norm_matrix  # shape: (n_orb, rank)

    def _time_reversal_separation(
        self, evecs_proj, rank=None, n_orb=None, *args, **kwargs
    ):
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
        # Quasi Wannier functions (qWF) by row
        # NOTE: we cannot directly split the qWFs by taking every other row, since they are distributed. We need to gather them first, then split them, and then distribute them again.
        revecs_orig = self.gather(evecs_proj, n_orb, rank)
        del evecs_proj

        revecs, sigma_y = None, None
        if self.is_master_rank:
            # take every other row of revecs_orig to get the vector of each time reversal pair
            revecs = revecs_orig[:, ::2]  # [n_orb,rank]

            pauli_y = np.array([[0, -1j], [1j, 0]])
            sigma_y = np.kron(np.eye(n_orb // 2), pauli_y)

        revecs = self.distribute(revecs)
        sigma_y = self.distribute(sigma_y)

        # Cycle over the number of degenerate subspaces
        tr_evecs = 1.0j * self.matmul(
            sigma_y, revecs.conj(), (n_orb, n_orb), (n_orb, rank // 2), "N", "N"
        )

        return revecs, tr_evecs  # shape: (n_orb, rank//2)
