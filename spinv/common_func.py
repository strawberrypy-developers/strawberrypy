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
    
def partialtrace(matrix, num : int):
    """
    Computes the partial trace of a given matrix, num specifies the number of elements to include in the partial trace
    """
    if not int(len(matrix) / num) == len(matrix) / num:
        raise RuntimeError("Number of element to take the partial trace does not divide the matrix dimension.")
    return np.array([np.sum([matrix[i * num + k, i * num + k] for k in range(num)]) for i in range(int(len(matrix) / num))])