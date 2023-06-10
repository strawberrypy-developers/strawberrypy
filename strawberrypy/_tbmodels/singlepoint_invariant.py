import numpy as np
import scipy.linalg as la
from .lattice import *
from strawberrypy.common_func import *

def solve_one(model, k):
    hamiltonian = model.hamilton(k,convention=1)

    eigval, eigvec = la.eigh(hamiltonian)

    return eigval, eigvec.T

def single_point_chern (model, formula='symmetric'):
    n_orb = len(model.pos)
    n_occ = model.occ

    # Gamma point is where to performe the single diagonalization
    point=[0.,0.]

    chern = []
    _, u_n0 = solve_one(model,point)

    b1, b2 = reciprocal_vec(model)
    u_nb1 = periodic_gauge(u_n0, b1, model)
    u_nb2 = periodic_gauge(u_n0, b2, model)

    udual_nb1 = dual_state(u_n0, u_nb1, n_occ, n_orb)
    udual_nb2 = dual_state(u_n0, u_nb2, n_occ, n_orb)

    if (formula=='asymmetric' or formula =='both'):
        sum_occ = 0.
        for i in range(n_occ):
            sum_occ += np.vdot(udual_nb1[i],udual_nb2[i])

        chern.append(-np.imag(sum_occ)/np.pi)

    if (formula=='symmetric' or formula =='both'):
        u_nmb1 = periodic_gauge(u_n0, -b1, model)
        u_nmb2 = periodic_gauge(u_n0, -b2, model)

        udual_nmb1 = dual_state(u_n0, u_nmb1, n_occ, n_orb)
        udual_nmb2 = dual_state(u_n0, u_nmb2, n_occ, n_orb)

        sum_occ = 0.
        for i in range(n_occ):
            sum_occ += np.vdot((udual_nmb1[i]-udual_nb1[i]),(udual_nmb2[i]-udual_nb2[i]))

        chern.append(-np.imag(sum_occ)/(4*np.pi) )

    return chern

def single_point_spin_chern(model, spin='down', formula='symmetric', return_gap_pszp=False):
    n_orb = len(model.pos)
    n_occ = model.occ
    n_sub = n_occ//2

    # Gamma point is where to performe the single diagonalization
    point=[0.,0.]

    closing_gap = False
    spin_chern = []

    b1, b2 = reciprocal_vec(model)

    _, u_n0 = solve_one(model,point)
    
    pszp = pszp_matrix(u_n0, n_occ)
    eval_pszp, eig_pszp = la.eigh(pszp)
    gap_pszp = eval_pszp[n_sub] - eval_pszp[n_sub-1]
    eig_pszp = eig_pszp.T

    if gap_pszp<10**(-14):
        closing_gap = True

    if closing_gap==True:
        raise Exception('Closing PszP gap!!')
    else:
        #check symmetry of P Sz P spectrum 
        if (eval_pszp[n_sub]*eval_pszp[n_sub-1]>0):
            raise Exception('P Sz P spectrum NOT symmetric!!!')

        q_0 = np.zeros([n_occ, n_orb], dtype=complex)
        q_0 = eig_pszp @ u_n0[:n_occ,:]

        q_b1 = periodic_gauge(q_0, b1, model)
        q_b2 = periodic_gauge(q_0, b2, model)

        qdual_b1 = dual_state_spin(q_0, q_b1, n_sub, n_orb, spin)
        qdual_b2 = dual_state_spin(q_0, q_b2, n_sub, n_orb, spin)

        if (formula=='asymmetric' or formula=='both'):
            sum_occ = 0.
            for i in range(n_sub):
                sum_occ += np.vdot(qdual_b1[i],qdual_b2[i])

            spin_chern.append(-np.imag(sum_occ)/np.pi)


        if (formula=='symmetric' or formula=='both'):
            q_mb1 = periodic_gauge(q_0,-b1,model)
            q_mb2 = periodic_gauge(q_0,-b2,model)

            qdual_mb1 = dual_state_spin(q_0, q_mb1, n_sub, n_orb, spin)
            qdual_mb2 = dual_state_spin(q_0, q_mb2, n_sub, n_orb, spin)

            sum_occ = 0.
            for i in range(n_sub):
                sum_occ += np.vdot((qdual_mb1[i]-qdual_b1[i]),(qdual_mb2[i]-qdual_b2[i]))

            spin_chern.append(-np.imag(sum_occ)/(4*np.pi)) 

    if return_gap_pszp==True:
        spin_chern.append(gap_pszp)
   
    return spin_chern
