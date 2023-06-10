import numpy as np
import random
import scipy.linalg as la
import time

def dual_state(un0, unb, n_occ, n_orb):
    s_matrix_b = np.zeros([n_occ,n_occ], dtype=np.complex128)
    s_matrix_b = np.conjugate(un0[:n_occ,:]) @ (unb[:n_occ,:]).T
    
    s_inv_b = np.linalg.inv(s_matrix_b)

    udual_nb = np.zeros((n_occ, n_orb), dtype=np.complex128)
    udual_nb = (s_inv_b.T) @ unb[:n_occ,:]

    return udual_nb

def dual_state_spin(q0, qb, n_sub, n_orb, spin):

    s_matrix_b = np.zeros([n_sub,n_sub], dtype=np.complex128)
    udual_nb = np.zeros((n_sub, n_orb), dtype=np.complex128)

    if spin ==  'down':
        s_matrix_b = np.conjugate(q0[:n_sub,:]) @ (qb[:n_sub,:]).T
        s_inv_b = np.linalg.inv(s_matrix_b)
        udual_nb = (s_inv_b.T) @ qb[:n_sub,:]
    elif spin == 'up':
        s_matrix_b = np.conjugate(q0[n_sub:,:]) @ (qb[n_sub:,:]).T
        s_inv_b = np.linalg.inv(s_matrix_b)
        udual_nb = (s_inv_b.T) @ qb[n_sub:,:]
    return udual_nb

def pszp_matrix (u_n0, n_occ):
    sz = [1,-1]*n_occ
    sz_un0 = (sz*u_n0).T
    pszp = np.ndarray([n_occ,n_occ],dtype=complex)
    pszp = u_n0[:n_occ,:].conjugate() @ sz_un0[:,:n_occ]
    return pszp

def commutator(A, B):
    """
    Computes the commutator [A, B]
    """
    return np.array(A @ B - B @ A)

def xy_to_index(fixed_coordinate : str, xy : int, nx : int, ny : int, atoms_uc : int):
    """
    Returns the indices of all the sites (including indices of atoms within the untit cell) along fixed_coordinate starting from position xy

        Args:
        - fixed_coordinate : direction in which look for the indices, allowed values are 'x' and 'y'
        - xy : starting position
        - nx, ny : lattice dimension
        - atoms_uc : number of atoms in the unit cell

        Returns:
        - indices : the list of indiecs along the given direction
    """
    if fixed_coordinate == 'x':
        return np.array([atoms_uc * xy * ny + i for i in range(atoms_uc * ny)]).flatten().tolist()
    elif fixed_coordinate == 'y':
        return np.array([[atoms_uc * xy + atoms_uc * ny * i + j for j in range(atoms_uc)] for i in range(0, nx)]).flatten().tolist()
    else:
        raise RuntimeError("Direction not allowed, only 'x' or 'y'")
    
def partialtrace(matrix, num : int):
    """
    Computes the partial trace of a given matrix, num specifies the number of elements to include in the partial trace
    """
    if not int(len(matrix) / num) == len(matrix) / num:
        raise RuntimeError("Number of element to take the partial trace does not divide the matrix dimension.")
    return np.array([np.sum([matrix[i * num + k, i * num + k] for k in range(num)]) for i in range(int(len(matrix) / num))])

def lattice_contraction(xcoord : np.ndarray, ycoord : np.ndarray, cutoff : float, nx : int, ny : int, atoms_uc : int):
    """
    Define the sites in the lattice that have to be contracted for each point in order to evaluate averages

        Args:
        - xcoord, ycoord : coordinates of the sites
        - cutoff : maximum radius of the contraction window
        - nx, ny : dimensions of the lattice
        - atoms_uc : atoms in the unit cell

        Returns:
        - contraction : the list of sites that have to be contracted in a point to evaluate averages for each point
    """
    contraction = []
    def within_range(current, trial):
        return True if (xcoord[current] - xcoord[trial]) ** 2 + (ycoord[current] - ycoord[trial]) ** 2 < cutoff ** 2 else False

    for current in range(nx * ny):
        contraction.append(np.array([np.array([atoms_uc * trial + ind for ind in range(atoms_uc)], dtype = int) for trial in range(nx * ny) if within_range(atoms_uc * trial, atoms_uc * current)]).flatten())

    return contraction

def average_over_radius(vals, xcoord, ycoord, cutoff, nx, ny, atoms_uc, contraction = None):
    """
    Computes the average over a certain radius of vals in a lattice

        Args:
        - vals : the values to average
        - xoord, ycoord : coordinates of the sites
        - cutoff : maximum radius of the contraction window
        - nx, ny : dimensions of the lattice
        - atoms_uc : atoms in the unit cell
        - contraction : if the contraction window has already been calculated, it is possible to pass it, default is None and the contraction window is evaluated here

        Returns:
        - contracted_vals : values averaged using the contraction
    """
    return_vals = []

    if contraction is None:
        contraction = lattice_contraction(xcoord, ycoord, cutoff, nx, ny, atoms_uc)

    # Macroscopia average within a certain radius
    for current in range(nx * ny):
        tmp = [vals[ind] for ind in contraction[current]]
        return_vals.append(np.sum(tmp) / (len(tmp) / atoms_uc))

    # Sum the averages over the unit cell
    return np.array(return_vals)