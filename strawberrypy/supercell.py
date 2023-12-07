import numpy as np

import tbmodels as tbm
import pythtb as ptb

from .classes import Model

from . import _tbmodels
from . import _pythtb

import scipy.linalg as la
from opt_einsum import contract
from .utils import *

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


    def pszp_matrix(self, u_n0):
        sz = self.sz
        #sz_un0 = (sz@u_n0).T
        pszp = np.ndarray([self.n_occ,self.n_occ],dtype=complex)
        pszp = u_n0[:self.n_occ,:].conjugate() @ (sz @ u_n0[:self.n_occ,:].T)
        return pszp


    def periodic_gauge(self, u_n0, b, n_occ = None):
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


    def single_point_chern(self, formula : str = 'both', return_ham_gap : bool = False):

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
    

    def single_point_spin_chern(self, spin : str = 'down', formula : str = 'both', return_pszp_gap : bool = False, return_ham_gap : bool = False):
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

    def pbc_local_chern_marker(self, direction : int = None, start : int = 0, return_projector : bool = False, input_projector = None, formula : str = "symmetric", macroscopic_average : bool = False, cutoff : float = 0.8, smearing_temperature : float = 0, fermidirac_cutoff : float = 0.1):
        r"""
        Evaluate the local Chern marker on the whole supercell if ``direction`` is ``None``. If ``direction`` is not ``None`` evaluates the PBC local Chern marker along ``direction`` starting from ``start``. Allowed directions are ``0`` (meaning along :math:`\mathbf{a}_1`), and ``1`` (meaning along :math:`\mathbf{a}_2`).
        
        Parameters
        ----------
            direction :
                Direction along which to compute the PBC local Chern marker. Default is ``None`` (returns the marker on the whole supercell). Allowed directions are ``0`` (meaning along :math:`\mathbf{a}_1`), and ``1`` (meaning along :math:`\mathbf{a}_2`).
            start :
                If ``direction`` is not ``None``, is the coordinate of the unit cell from which the evaluation of the PBC local Chern marker starts. For instance, if interested on the value of the PBC local marker along the :math:`\mathbf{a}_1` direction at half height, it should be set ``direction = 0`` and ``start = Ly // 2``.
            return_projector :
                If ``True``, returns the ground state projector at the end of the calculation. Default is ``False``.
            input_projector :
                Input the list of projectors :math:`[\mathcal{P}_{\Gamma},\mathcal{P}_{\mathbf{b}_1},\mathcal{P}_{\mathbf{b}_2}]` (or :math:`[\mathcal{P}_{\Gamma},\mathcal{P}_{\mathbf{b}_1},\mathcal{P}_{\mathbf{b}_2},\mathcal{P}_{-\mathbf{b}_1},\mathcal{P}_{-\mathbf{b}_2}]` if ``formula == 'symmetric'``) to be used in the calculation. Default is ``None``, which means that it is computed from the model stored in the class.
            formula :
                Formula to be used. Default is ``'symmetric'``, which is computationally more demanding but converges faster. Any other input will result in the ``'asymetric'`` formulation.
            macroscopic_average :
                If ``True``, returns the PBC local Chern marker averaged in real space over a radius equal to ``cutoff``. Default is ``False``.
            cutoff :
                Cutoff set for the calculation of the macroscopic average in real space of the PBC local Chern marker.
            smearing_temperature :
                Set a fictitious temperature :math:`T_s` to be used when weighting the eigenstates of the Hamiltonian comprising the ground state projector. In particular, the ground state projector is computed as :math:`\mathcal P=\sum_{n}f(\epsilon_n, T_s, \mu)|u_n\rangle\langle u_n|` where :math:`f(\epsilon_n, T_s, \mu)` is the Fermi-Dirac distribution, :math:`\mu` is the chemical potential and :math:`\mathcal{H}_{\mathbf{k}}|u_n\rangle=\epsilon_n|u_n\rangle`. Introducing some smearing is particularly useful when dealing with heterojunctions o inhomogeneous models whose insulating gap is small in order to improve the convergence of the local marker. Default is ``0``, so no smearing is introduced and a model half-filled is implied.
            fermidirac_cutoff :
                Cutoff imposed on the Fermi-Dirac distribution to further improve the convergence, mostly when :math:`T_s\neq0`. Default is ``0.1``, which looks appropriate in most cases.

        Returns
        -------
            lattice_chern :
                PBC local Chern marker evaluated on the whole lattice if ``direction`` is ``None``.
            lcm_direction :
                PBC local Chern marker evaluated along ``direction`` starting from ``start``.
            return_proj :
                List of projectors :math:`[\mathcal{P}_{\Gamma},\mathcal{P}_{\mathbf{b}_1},\mathcal{P}_{\mathbf{b}_2}]` (or :math:`[\mathcal{P}_{\Gamma},\mathcal{P}_{\mathbf{b}_1},\mathcal{P}_{\mathbf{b}_2},\mathcal{P}_{-\mathbf{b}_1},\mathcal{P}_{-\mathbf{b}_2}]` if ``formula == 'symmetric'``) used in the calculation.
        """
        return_proj = []

        if input_projector is None:
            eigenvals, eigenvecs = la.eigh(self.hamiltonian)

            # Find the chemical potential
            mu = chemical_potential(eigenvals, smearing_temperature, self.n_occ)

            # Evaluate the effective number of occupied states whose occupation is greater than the Fermi-Dirac cutoff
            rank = np.sum(fermidirac(eigenvals, smearing_temperature, mu) > fermidirac_cutoff)

            eigenvecs_use = eigenvecs.T

            # Reciprocal lattice vectors
            b1, b2 = self.reciprocal_vec()

            # Periodic gauge along b_1 and b_2
            u_nb1 = self.periodic_gauge(eigenvecs_use, b1, n_occ = rank)
            u_nb2 = self.periodic_gauge(eigenvecs_use, b2, n_occ = rank)

            # Dual states at b_1 and b_2
            udual_b1 = self.dual_state(eigenvecs_use, u_nb1, n_sub = rank)
            udual_b2 = self.dual_state(eigenvecs_use, u_nb2, n_sub = rank)
        
            # Evaulate the ground state projector, and projectors P_b1 and P_b2
            gsp = contract("ij,ik->jk", eigenvecs_use[:rank, :], (fermidirac(eigenvals[:rank], smearing_temperature, mu) * eigenvecs_use[:rank, :].conjugate().T).T)
            pb1 = contract("ij,ik->jk", udual_b1, (fermidirac(eigenvals[:rank], smearing_temperature, mu) * (udual_b1.conjugate().T)).T)
            pb2 = contract("ij,ik->jk", udual_b2, (fermidirac(eigenvals[:rank], smearing_temperature, mu) * (udual_b2.conjugate().T)).T)
            return_proj.append(gsp); return_proj.append(pb1); return_proj.append(pb2)
            p = pb1 @ pb2 - pb2 @ pb1

            # If I want the symmetric formula I need to do the same also for -b_1 and -b_2
            if formula == "symmetric":
                u_nmb1 = self.periodic_gauge(eigenvecs_use, -1 * b1, n_occ = rank)
                u_nmb2 = self.periodic_gauge(eigenvecs_use, -1 * b2, n_occ = rank)

                udual_mb1 = self.dual_state(eigenvecs_use, u_nmb1, n_sub = rank)
                udual_mb2 = self.dual_state(eigenvecs_use, u_nmb2, n_sub = rank)

                pmb1 = contract("ij,ik->jk", udual_mb1, (fermidirac(eigenvals[:rank], smearing_temperature, mu) * (udual_mb1.conjugate().T)).T)
                pmb2 = contract("ij,ik->jk", udual_mb2, (fermidirac(eigenvals[:rank], smearing_temperature, mu) * (udual_mb2.conjugate().T)).T)
                return_proj.append(pmb1)
                return_proj.append(pmb2)

                p += ( (pmb1 @ pmb2 - pmb2 @ pmb1) - (pb1 @ pmb2 - pmb2 @ pb1) - (pmb1 @ pb2 - pb2 @ pmb1) )
        else:
            gsp = input_projector[0]
            pb1 = input_projector[1]
            pb2 = input_projector[2]
            p = pb1 @ pb2 - pb2 @ pb1

            if formula == 'symmetric':
                pmb1 = input_projector[3]
                pmb2 = input_projector[4]
                p += ( (pmb1 @ pmb2 - pmb2 @ pmb1) - (pb1 @ pmb2 - pmb2 @ pb1) - (pmb1 @ pb2 - pb2 @ pmb1) )

        # PBC Chern marker operator
        chern_operator = -np.imag(p @ gsp) * float((self.Lx * self.Ly) / (np.pi * (8 if formula == "symmetric" else 2)))

        # If macroscopic_average I have to compute the lattice values with the averages first (explicit PBC contraction passed)
        if macroscopic_average or self.disordered:
            contraction = self._PBC_lattice_contraction(cutoff)
            pbclcm = self._average_over_radius(np.diag(chern_operator), cutoff, contraction = contraction)

        if direction is not None:
            # Evaluate index of the selected direction
            indices = self._xy_to_index('x' if direction == 1 else 'y', start)

            # If macroscopic_average consider the averaged lattice, else the Chern operators
            if macroscopic_average or self.disordered:
                pbclcm_line = [pbclcm[int(indices[self.states_uc * i] / self.states_uc)] for i in range(int(len(indices) / self.states_uc))]
            else:
                pbclcm_line = [np.sum([chern_operator[indices[self.states_uc * i + j], indices[self.states_uc * i + j]] for j in range(self.states_uc)]) for i in range(int(len(indices) / self.states_uc))]
            
            if not return_projector:
                return np.array(pbclcm_line)
            else:
                return np.array(pbclcm_line), np.array(return_proj)

        if not macroscopic_average and not self.disordered:
            pbclcm = [np.sum([chern_operator[self.states_uc * i + j, self.states_uc * i + j] for j in range(self.
            states_uc)]) for i in range(int(len(chern_operator) / self.states_uc))]

            # Repeat to ensure that the dimension of the marker matches the dimension of the position matrices, since if not macroscopic_average the value of the marker is defined per unit cell
            pbclcm = np.repeat(pbclcm, self.states_uc)
        
        if not return_projector:
            return np.array(pbclcm)
        else:
            return np.array(pbclcm), np.array(return_proj)