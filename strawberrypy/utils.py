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
    

def chemical_potential(evals, temperature : float, occupied_states : int):
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

    Returns
    -------
        mu : :python:`float`
            The chemical potential of the system.

    .. warning::
        The chemical potential is calculated via bisection method, if convergence is not achieved in 200 iterations, an error is returned.
    """
    mu_min = np.min(evals)
    mu_max = np.max(evals)
    mu = 0
    niter = 0
    maxiter = 200
    
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
    

def unique_vacancies(num : int, Lx : int, Ly : int, basis : int, seed : int = None):
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

    # Generate num unique entries
    while len(unique_list) < num:
        # Trial entry
        trial = [np.random.randint(Lx), np.random.randint(Ly), np.random.randint(basis)]
        
        # Generate internal index
        trial_index = Ly * basis * trial[0] + basis * trial[1] + trial[2]

        # If this is a new entry store it
        if not trial_index in indexes:
            unique_list.append(trial)

    return unique_list