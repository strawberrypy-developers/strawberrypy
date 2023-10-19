import numpy as np

import tbmodels as tbm
import pythtb as ptb

from .classes import Model

from . import _tbmodels
from . import _pythtb

import scipy.linalg as la
from opt_einsum import contract

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


    def periodic_gauge (self, u_n0, b, n_occ = None):
        """
        Returns the matrix of occupied eigenvectors at the edge of the Brillouin zone imposing periodic gauge upon the eigenvectors at Gamma
        """
        if n_occ is None: n_occ = self.n_occ

        orb_c = self.cart_positions
        vec_scal_b = orb_c @ b
        vec_exp_b = np.exp(-1.j*vec_scal_b)

        u_nb = vec_exp_b.T * u_n0[:n_occ,:]

        return u_nb


    def dual_state(self, un0, unb, spin = None, n_sub = None):

        if n_sub is None:
            if self.spinful: 
                n_sub = self.n_occ//2
            else:
                n_sub = self.n_occ

        s_matrix_b = np.zeros([n_sub,n_sub], dtype=np.complex128)
        udual_nb = np.zeros((n_sub, self.n_orb), dtype=np.complex128)

        if (spin == 'down' or spin == None):
            s_matrix_b = np.conjugate(un0[:n_sub,:]) @ (unb[:n_sub,:]).T
            s_inv_b = np.linalg.pinv(s_matrix_b)
            udual_nb = (s_inv_b.T) @ unb[:n_sub,:]
        elif spin == 'up':
            s_matrix_b = np.conjugate(un0[n_sub:,:]) @ (unb[n_sub:,:]).T
            s_inv_b = np.linalg.pinv(s_matrix_b)
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

    #################################################
    # Local PBC marker
    #################################################

    def pbc_local_chern_marker(self, direction : int = None, start : int = 0, return_projector : bool = False, input_projector = None, formula : str = 'symmetric', smearing_temperature : float = 0, fermidirac_cutoff : float = 0.1, macroscopic_average : bool = False, cutoff : float = 0.8):
        """
        Evaluate the PBC local Chern marker on the whole lattice if direction is None. If direction is not None evaluates the Chern marker along direction starting from start.
        
        Args:
            - direction : direction along which compute the PBC local Chern marker, default is None (returns the marker on the whole lattice), allowed values are 0 for 'x' direction and 1 for 'y' direction
            - start : if direction is not None, is the coordinate of the unit cell at the start of the evaluation of the PBC Chern marker
            - return_projector : if True, returns the ground state projector at the end of the calculation, default is False
            - input_projector : input the list of projectors [P, P1, P2] (or [P, P1, P2, P-1, P-2] if formula is symmetric) used in the calcualtions. Default is None, which means they are computed from the model
            - formula : 'symmetric' or 'asymmetric' PBC local Chern marker formula
            - smearing_temperature : if smearing is needed, the temperature to use in the Fermi-Dirac distribution
            - fermidirac_cutoff : if smearing is introduced, the cutoff on the occupied states for the Fermi-Dirac distribution
            - macroscopic_average : if True, returns the PBC local Chern marker averaged over a radius equal to the cutoff
            - cutoff : cutoff set for the calculation of averages

        Returns:
            - lattice_chern : local Chern marker of the whole lattice if direction is None
            - lcm_direction : local Chern marker along direction starting from start
            - projector : a list of prjectors [P, P1, P2] (or [P, P1, P2, P-1, P-2] if formula is symmetric), returned if return_projector is set True (default is False)
        """
        
        # Auxiliary functions
        def fermidirac(e, t, mu):
            if t == 0:
                out = []
                for en in e:
                    out.append(1 if en < mu else 0)
                return np.array(out)
            else:
                return 1 / ( 1 + np.exp((e - mu) / t) )
        
        return_proj = []

        if input_projector is None:
            eigenvals, eigenvecs = la.eigh(self.hamiltonian)

            # Find the chemical potential
            mu_min = np.min(eigenvals); mu_max = np.max(eigenvals); mu = 0
            numatoms = self.n_occ; niter = 0; maxiter = 200
            while True:
                mu = 0.5 * (mu_min + mu_max)
                n_exp = np.sum(fermidirac(eigenvals, smearing_temperature, mu))

                if n_exp < numatoms:
                    mu_min = mu
                else:
                    mu_max = mu
                
                if np.abs(n_exp - numatoms) < 1e-6:
                    break

                niter += 1
                if niter > maxiter:
                    raise RuntimeError("Bisection method did not converge")

            nocc = np.sum(fermidirac(eigenvals, smearing_temperature, mu) > fermidirac_cutoff)

            # Periodic gauge
            b1, b2 = self.reciprocal_vec()
            eigenvecs_use = eigenvecs.T

            u_nb1 = self.periodic_gauge(eigenvecs_use, b1)
            u_nb2 = self.periodic_gauge(eigenvecs_use, b2)

            udual_b1 = self.dual_state(eigenvecs_use, u_nb1)
            udual_b2 = self.dual_state(eigenvecs_use, u_nb2)

            gsp = contract("ij,ik->jk", eigenvecs_use[:nocc, :], (fermidirac(eigenvals[:nocc], smearing_temperature, mu) * eigenvecs_use[:nocc, :].conjugate().T).T)
            pb1 = contract("ij,ik->jk", udual_b1, (fermidirac(eigenvals[:nocc], smearing_temperature, mu) * udual_b1.conjugate().T).T)
            pb2 = contract("ij,ik->jk", udual_b2, (fermidirac(eigenvals[:nocc], smearing_temperature, mu) * udual_b2.conjugate().T).T)
            return_proj.append(gsp); return_proj.append(pb1); return_proj.append(pb2)
            p = pb1 @ pb2 - pb2 @ pb1

            if formula == "symmetric":
                u_nmb1 = self.periodic_gauge(eigenvecs_use, -b1)
                u_nmb2 = self.periodic_gauge(eigenvecs_use, -b2)

                udual_mb1 = self.dual_state(eigenvecs_use, u_nmb1)
                udual_mb2 = self.dual_state(eigenvecs_use, u_nmb2)

                pmb1 = contract("ij,ik->jk", udual_mb1, (fermidirac(eigenvals[:nocc], smearing_temperature, mu) * udual_mb1.conjugate().T).T)
                pmb2 = contract("ij,ik->jk", udual_mb2, (fermidirac(eigenvals[:nocc], smearing_temperature, mu) * udual_mb2.conjugate().T).T)
                return_proj.append(pmb1)
                return_proj.append(pmb2)

                p += ( (pmb1 @ pmb2 - pmb2 @ pmb1) - (pb1 @ pmb2 - pmb2 @ pb1) - (pmb1 @ pb2 - pb2 @ pmb1) )
        else:
            gsp = input_projector[0]
            pb1 = input_projector[1]
            pb1 = input_projector[2]
            p = pb1 @ pb2 - pb2 @ pb1

            if formula == 'symmetric':
                pmb1 = input_projector[3]
                pmb2 = input_projector[4]
                p += ( (pmb1 @ pmb2 - pmb2 @ pmb1) - (pb1 @ pmb2 - pmb2 @ pb1) - (pmb1 @ pb2 - pb2 @ pmb1) )

        chern_operator = -np.imag(p @ gsp) * float((self.Lx * self.Ly) / (np.pi * (8 if formula == "symmetric" else 2)))

        if macroscopic_average or self.disordered:
            contraction = self._PBC_lattice_contraction(cutoff)
            pbclcm = self._average_over_radius(np.diag(chern_operator), cutoff, contraction = contraction)

        if direction is not None:
            # Evaluate index of the selected direction
            indices = self._xy_to_index('x' if direction == 1 else 'y', start)

            # If macroscopic average consider the averaged lattice, else the Chern operators
            if macroscopic_average or self.disordered:
                pbclcm_line = [pbclcm[int(indices[self.states_uc * i] / self.states_uc)] for i in range(int(len(indices) / self.states_uc))]
            else:
                pbclcm_line = [np.sum([chern_operator[indices[self.states_uc * i + j], indices[self.states_uc * i + j]] for j in range(self.states_uc)]) for i in range(int(len(indices) / self.states_uc))]
            
            if not return_projector:
                return np.array(pbclcm_line)
            else:
                return np.array(pbclcm_line), np.array(return_proj)

        if not macroscopic_average and not self.disordered:
            pbclcm = [np.sum([chern_operator[self.states_uc * i + j, self.states_uc * i + j] for j in range(self.states_uc)]) for i in range(int(len(chern_operator) / self.states_uc))]
            pbclcm = np.repeat(pbclcm, self.states_uc)
        
        if not return_projector:
            return np.array(pbclcm)
        else:
            return np.array(pbclcm), np.array(return_proj)