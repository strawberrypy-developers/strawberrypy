import numpy as np

def fermidirac(evals, temperature : float, mu : float):
    r"""
    The Fermi-Dirac distribution :math:`f(\epsilon, T, \mu)=\big[ 1 + e^{\frac{\epsilon-\mu}{T}} \big]^{-1}`.
    
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
            The occupation(s) of the state corresponding to the given energy(ies)
    """
    if temperature < 1e-6:
        if evals.shape == ():
            return 1 if evals <+ mu else 0
        else:
            return np.array([ 1 if e <= mu else 0 for e in evals ])
    else:
        return 1 / (1 + np.exp( (evals - mu) / temperature ))
    

def chemical_potential(evals, temperature : float, occupied_states : int, **kwargs):
    r"""
    Calculate the chemical potential of a given model knowing the eigenvalue distribution and the number of electrons (occupied states) in the system.

    Parameters
    ----------
        evals :
            List of eigenvalues of the Hamiltonian.
        temperature :
            Temperature (real or fictitious, as in the case of smearing) of the system, appearing in the Fermi-Dirac distribution.
        occupied_states :
            Number of occupied states of the system.
        kwargs :
            Additional keyword arguments, such as:
            
            - `maxiter` : Maximum number of iterations for the bisection method to find the chemical potential. Default is 200.

    Returns
    -------
        mu : :python:`float`
            The chemical potential of the system.
    """
    mu_min = np.min(evals)
    mu_max = np.max(evals)
    mu = 0
    niter = 0
    maxiter = kwargs.get('maxiter', 200)
    
    while True:
        mu = 0.5 * (mu_min + mu_max)
        n_exp = np.sum(fermidirac(evals, temperature, mu))

        if n_exp < occupied_states:
            mu_min = mu
        else:
            mu_max = mu
        
        if np.abs(n_exp - occupied_states) < 1e-6:
            break

        niter += 1
        if niter > maxiter:
            raise RuntimeError("Chemical potential cannot be found: bisection method failed (200 iterations)")
    return mu


def smearing(vecs, gamma_hevecs, evals, temperature : float, mu : float):
    r"""
    Smearing introduced to improve the convergence of the formula: it measures how much the projector built from :python:`vecs` is similar to the one built from :python:`gamma_hevecs`, the eigenstates of the Hamiltonian at the :math:`\Gamma`-point. Naming :math:`|\phi_n\rangle` the vectors in :python:`vecs` and :math:`|u_n\rangle` the ones in :python:`gamma_hevecs`, the smearing factor is computed as :math:`c_n=\sum_m f(\epsilon_m, T_s, \mu)|\langle \phi_n|u_m\rangle|^2`, where :math:`\epsilon_m` is the eigenvalue corresponding to the eigenstate :math:`|u_m\rangle`, :math:`T_s` is the smearing temperature, :math:`\mu` is the chemical potential.

    Parameters
    ----------
        vecs :
            List of states that need to be weighted according to some smearing.
        gamma_hevecs :
            Eigenstates of th Hamiltonian at the :math:`\Gamma`-point.
        evals :
            Eigenvales of the Hamiltonian at the :math:`\Gamma`-point.
        temperature :
            Temperature introduced to smoothen the occupation of the states (smearing temperature).
        mu :
            Chemical potential of the system.

    Returns
    -------
        smearing_coeffs : :python:`np.array`
            A list of smearing coefficients that weights the states :python:`vecs`.
    """
    if temperature < 1e-6:
        return np.array([1 for _ in range(vecs.shape[1])])
    else:
        return np.array([np.sum([fermidirac(evals[m], temperature, mu) * np.abs(np.vdot(vecs[:, n], gamma_hevecs[:, m])) ** 2 for m in range(gamma_hevecs.shape[1])]) for n in range(vecs.shape[1])])
    

def unique_vacancies(num : int, Lx : int, Ly : int, basis : int, atom_type : int = None, same_number : bool = False, seed : int = None):
    r"""
    Returns a list of unique random lattice sites to be removed in the model using the method :python:`add_vacancies`.

    Parameters
    ----------
        num :
            Number of lattice positions to generate.
        Lx :
            Number of unit cells of the model along the :math:`\mathbf{a}_1` direction.
        Ly :
            Number of unit cells of the model along the :math:`\mathbf{a}_2` direction.
        basis :
            Number of atoms per unit cell.
        atom_type :
            Index of the atom to be removed.
        same_number :
            If :python:`True`, the same number of each type of atom is removed.
        seed :
            Seed for the random number generation. Default is :python:`None`.

    Returns
    -------
        unique_list : :python:`list`
            List of unique random lattice site.
    """

    indexes = []
    unique_list = []

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    if same_number:
        atom = np.zeros(num, dtype=np.integer)
        type_B = np.zeros(num, dtype=np.integer)
        type_B[:num//basis] = np.ones(num//basis, dtype=np.integer)

        nonzeros = np.sort(np.nonzero(np.random.permutation(type_B))[0])
        atom[nonzeros] = np.ones(num//basis, dtype=np.integer)

        ind = 0
        # Generate num unique entries
        while len(unique_list) < num:
            # Trial entry
            at_type = atom[ind]

            trial = [np.random.randint(Lx), np.random.randint(Ly), at_type]
            
            # Generate internal index
            trial_index = Ly * basis * trial[0] + basis * trial[1] + trial[2]

            # If this is a new entry store it
            if not trial_index in indexes:
                indexes.append(trial_index)
                unique_list.append(trial)

                ind += 1
    else:
        # Generate num unique entries
        while len(unique_list) < num:
            # Trial entry
            if atom_type is not None:
                trial = [np.random.randint(Lx), np.random.randint(Ly), atom_type]
            else:
                trial = [np.random.randint(Lx), np.random.randint(Ly), np.random.randint(basis)]
            
            # Generate internal index
            trial_index = Ly * basis * trial[0] + basis * trial[1] + trial[2]

            # If this is a new entry store it
            if not trial_index in indexes:
                indexes.append(trial_index)
                unique_list.append(trial)

    return unique_list


def _add_occupied_states(evals, fd_cut, nocc_t0, ntba, **kwargs):
    r"""
    Functionality to determinate the smearing temperature and the chemical potential such that the number of occupied states is nocc_t0 + ntba.

    Parameters
    ----------
        evals :
            Eigenvales of the Hamiltonian at the :math:`\Gamma`-point.
        fd_cut :
            Cutoff imposed on the Fermi-Dirac distribution.
        nocc_t0 :
            Number of occupied states at zero temperature.
        ntba :
            Number of states to be added relative to the a priori set filling at zero temperature.
        
        kwargs :
            Additional keyword arguments, such as:
            - `T_min` : Smallest temperature for the bisection procedure. Default is 0.
            - `T_max` : Highest temperature for the bisection procedure. Default is 0.1.
            - `max_iter` : Maximum number of iterations for the bisection method to find the chemical potential. Default is 200.

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
        return nocc_t0, 0.0, chemical_potential(evals, 0.0, nocc_t0)
    else :
        while True:
            temperature = 0.5 * (tmin + tmax)
            mu = chemical_potential(evals, temperature, nocc_t0)
            nocc = np.sum(fermidirac(evals, temperature, mu) > fd_cut)
            if nocc < nocc_t0 + ntba:
                tmin = temperature
            else:
                tmax = temperature
            if np.abs(nocc - nocc_t0 - ntba) < 1e-6:
                break
            niter += 1
            if niter > maxiter:
                print("Max iterations reached: leaving default values")
                return nocc_t0, 0.0, chemical_potential(evals, 0.0, nocc_t0)
        return nocc, temperature, mu


def _add_states_until_gap(evals, nocc_t0, gap_tol, fd_cut, **kwargs):
    r"""
        Functionality to determinate the smallest number of states to be added to nocc_t0 such that the system has a gap larger than gap_tol.

        Parameters
        ----------
            evals :
                Eigenvales of the Hamiltonian at the :math:`\Gamma`-point.
            nocc_t0 :
                Number of occupied states at zero temperature.
            gap_tol :
                Minimum value of the gap when adding states to :python:`nocc_t0`. Default is :python:`1e-8`.
            fd_cut :
                Cutoff imposed on the Fermi-Dirac distribution.

            kwargs :
                Additional keyword arguments, such as:
                - `T_min` : Smallest temperature for the bisection procedure. Default is 0.
                - `T_max` : Highest temperature for the bisection procedure. Default is 0.1.
                - `max_iter` : Maximum number of iterations for the bisection method to find the chemical potential. Default is 200.

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
        gap = evals[nocc_t0+i] - evals[nocc_t0+i-1]
        if gap < gap_tol:
            continue
        else:
            n_tba = i
            break

    nocc, temperature, mu = _add_occupied_states(evals, fd_cut, nocc_t0, ntba=n_tba, **kwargs)
    temp = np.linspace(0.0,temperature,101)
    t_f = temperature
    mu_f = mu
    for t in reversed(temp):
        mu_t = chemical_potential(evals, t, nocc_t0)
        nocc_t = np.sum(fermidirac(evals, t, mu_t) > fd_cut)
        if nocc_t < nocc:
            break
        else:
            t_f = t
            mu_f = mu_t

    print("ADD OCC STATES:\nn_states={0}\ntemperature_min={1}\nmu={2}\ngap={3}".format(nocc, t_f, mu_f,gap))
    return nocc, t_f, mu_f