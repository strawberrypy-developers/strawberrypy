import numpy as np

import tbmodels as tbm
import pythtb as ptb

from .classes import Model

from . import _tbmodels
from . import _pythtb

import scipy.linalg as la

class Supercell(Model):
    def __init__(self, tbmodel, Lx : int, Ly : int, spinful : bool = False):
        self.Lx = Lx
        self.Ly = Ly

        self.model = self._make_supercell(tbmodel)

        super().__init__(tbmodel = self.model, 
                         spinful = spinful,
                         states_uc = super()._calc_states_uc(tbmodel, spinful),
                         Lx = self.Lx,
                         Ly = self.Ly
                        )

    def _make_supercell(self, tbmodel):
        if (self.Lx != 1 and self.Ly != 1):
            if isinstance(tbmodel, tbm.Model):
                return tbmodel.supercell([self.Lx,self.Ly])
            elif isinstance(tbmodel, ptb.tb_model):
                return tbmodel.make_supercell([[self.Lx,0],[0,self.Ly]])
            else:
                raise NotImplementedError("Invalid model instance.")
        else:
            return tbmodel
        
    def reciprocal_vec(self):
        if isinstance(self.model, tbm.Model):
                return _tbmodels._reciprocal_vec(self.model)
        elif isinstance(self.model, ptb.tb_model):
            return _pythtb._reciprocal_vec(self.model)
        else:
            raise NotImplementedError("Invalid model instance.")
        
    def pszp_matrix (self, u_n0):
        sz = self.sz
        #sz_un0 = (sz@u_n0).T
        pszp = np.ndarray([self.n_occ,self.n_occ],dtype=complex)
        pszp = u_n0[:self.n_occ,:].conjugate() @ (sz @ u_n0[:self.n_occ,:].T)
        return pszp
        
    def periodic_gauge (self, u_n0, b):
        """
        Returns the matrix of occupied eigenvectors at the edge of the Brillouin zone imposing periodic gauge upon the eigenvectors at Gamma
        """
        n_occ = self.n_occ

        orb_c = self.cart_positions
        vec_scal_b = orb_c @ b
        vec_exp_b = np.exp(-1.j*vec_scal_b)

        u_nb = vec_exp_b.T * u_n0[:n_occ,:]

        return u_nb
        
    def dual_state(self, un0, unb, spin = None):

        if self.spinful: 
            n_sub = self.n_occ//2
        else:
            n_sub = self.n_occ

        s_matrix_b = np.zeros([n_sub,n_sub], dtype=np.complex128)
        udual_nb = np.zeros((n_sub, self.n_orb), dtype=np.complex128)

        if (spin == 'down' or spin == None):
            s_matrix_b = np.conjugate(un0[:n_sub,:]) @ (unb[:n_sub,:]).T
            s_inv_b = np.linalg.inv(s_matrix_b)
            udual_nb = (s_inv_b.T) @ unb[:n_sub,:]
        elif spin == 'up':
            s_matrix_b = np.conjugate(un0[n_sub:,:]) @ (unb[n_sub:,:]).T
            s_inv_b = np.linalg.inv(s_matrix_b)
            udual_nb = (s_inv_b.T) @ unb[n_sub:,:]

        return udual_nb

    def single_point_chern(self, formula, return_ham_gap=False):

        eig, u_n0 = la.eigh(self.hamiltonian)
        u_n0 = u_n0.T

        b1, b2 = self.reciprocal_vec()
        u_nb1 = self.periodic_gauge(u_n0, b1)
        u_nb2 = self.periodic_gauge(u_n0, b2)

        udual_nb1 = self.dual_state(u_n0, u_nb1)
        udual_nb2 = self.dual_state(u_n0, u_nb2)

        chern = {}

        if (formula=='asymmetric' or formula =='both'):
            sum_occ = 0.
            for i in range(self.n_occ):
                sum_occ += np.vdot(udual_nb1[i],udual_nb2[i])

            chern['asymmetric'] = -np.imag(sum_occ)/np.pi

        if (formula=='symmetric' or formula =='both'):
            u_nmb1 = self.periodic_gauge(u_n0, -b1)
            u_nmb2 = self.periodic_gauge(u_n0, -b2)

            udual_nmb1 = self.dual_state(u_n0, u_nmb1)
            udual_nmb2 = self.dual_state(u_n0, u_nmb2)

            sum_occ = 0.
            for i in range(self.n_occ):
                sum_occ += np.vdot((udual_nmb1[i]-udual_nb1[i]),(udual_nmb2[i]-udual_nb2[i]))

            chern['symmetric'] = -np.imag(sum_occ)/(4*np.pi)

        if return_ham_gap:
            chern['hamiltonian_gap'] = eig[self.n_occ] - eig[self.n_occ-1]

        return chern
    
    def single_point_spin_chern(self, spin='down', formula='both', return_pszp_gap=False, return_ham_gap=False):
        n_sub = self.n_occ//2

        eig, u_n0 = la.eigh(self.hamiltonian)
        u_n0 = u_n0.T

        pszp = self.pszp_matrix(u_n0)
        eval_pszp, eig_pszp = la.eigh(pszp)
        eig_pszp = eig_pszp.T
        gap_pszp = eval_pszp[n_sub] - eval_pszp[n_sub-1]

        spin_chern = {}

        if (gap_pszp < 10**(-14)):
            raise Exception('Closing PszP gap!!')
        elif (eval_pszp[n_sub]*eval_pszp[n_sub-1]>0) :
            #check symmetry of P Sz P spectrum 
            raise Exception('P Sz P spectrum NOT symmetric!!!')
        else :
            q_0 = np.zeros([self.n_occ, self.n_orb], dtype=complex)
            q_0 = eig_pszp @ u_n0[:self.n_occ,:]

            b1, b2 = self.reciprocal_vec()
            q_b1 = self.periodic_gauge(q_0, b1)
            q_b2 = self.periodic_gauge(q_0, b2)

            qdual_b1 = self.dual_state(q_0, q_b1, spin)
            qdual_b2 = self.dual_state(q_0, q_b2, spin)

        if (formula=='asymmetric' or formula=='both'):
            sum_occ = 0.
            for i in range(n_sub):
                sum_occ += np.vdot(qdual_b1[i],qdual_b2[i])

            spin_chern['asymmetric'] = -np.imag(sum_occ)/np.pi


        if (formula=='symmetric' or formula=='both'):
            q_mb1 = self.periodic_gauge(q_0,-b1)
            q_mb2 = self.periodic_gauge(q_0,-b2)

            qdual_mb1 = self.dual_state(q_0, q_mb1, spin)
            qdual_mb2 = self.dual_state(q_0, q_mb2, spin)

            sum_occ = 0.
            for i in range(n_sub):
                sum_occ += np.vdot((qdual_mb1[i]-qdual_b1[i]),(qdual_mb2[i]-qdual_b2[i]))

            spin_chern['symmetric'] = -np.imag(sum_occ)/(4*np.pi)

        if return_pszp_gap:
            spin_chern['pszp_gap'] = gap_pszp

        if return_ham_gap:
            spin_chern['hamiltonian_gap'] = eig[self.n_occ] - eig[self.n_occ-1]
   
        return spin_chern

        
        
    

        