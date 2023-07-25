import numpy as np
import argh
from spinv import classes, supercell
from spinv.example_models import kane_mele_tbmodels, kane_mele_pythtb, haldane_pythtb, haldane_tbmodels

import time

import scipy.linalg as la

def test_km(L=24, r=1., e=3., spin_o=0.3, w=2.0, spin_chern='down', which_formula='symmetric'):

    t1 = time.time()

    #create Kane-Mele model in supercell LxL through PythTB package
    km_pytb = kane_mele_pythtb(r,e,spin_o,L=1)

    #create Kane-Mele model in supercell LxL through TBmodels package
    km_tbm = kane_mele_tbmodels(r,e,spin_o,L=1)

    system_tbm = supercell.Supercell(km_tbm, Lx=L, Ly=L, spinful=True)
    system_pytb = supercell.Supercell(km_pytb, Lx=L, Ly=L, spinful=True)

    system_tbm.add_onsite_disorder(w,seed=10)
    system_pytb.add_onsite_disorder(w,seed=10)
    
    print(system_tbm.single_point_spin_chern(formula='both', return_pszp_gap = True, return_ham_gap=True))
    print(system_pytb.single_point_spin_chern(formula='both', return_pszp_gap = True, return_ham_gap=True))

    t2 = time.time()

    print(t2-t1)
    exit()
    print(system_tb.hamiltonian)

    print(system_tb.hamiltonian)

    system_tb.add_onsite_disorder(w, seed=10)
    system_py.add_onsite_disorder(w, seed=10)


def test_haldane(L=1, t=-4., t2=1., delta=2., pi_phi=-2., w=1.5, which_formula = 'symmetric'):

    #Haldane model parameters
    phi = np.pi/pi_phi

    #create Haldane model in supercell LxL through PythTB package
    h_pythtb_model = haldane_pythtb(delta, t, t2, phi, L)

    #create Haldane model in supercell LxL through TBmodels package
    h_tbmodels_model = haldane_tbmodels(delta, t, t2, phi, L)

    system_tbm = supercell.Supercell(h_tbmodels_model, Lx=3, Ly=3, spinful=False)
    system_pytb = supercell.Supercell(h_pythtb_model, Lx=3, Ly=3, spinful=False)

    print(system_tbm.singlepoint_chern(formula='both', return_ham_gap=True))
    print(system_pytb.singlepoint_chern(formula='both', return_ham_gap=True))

    exit()

    system_tb = classes.Model(h_tbmodels_model,spinful=False)
    system_py = classes.Model(h_pythtb_model,spinful=False)

    system_tb.add_onsite_disorder(w, seed=10)
    system_py.add_onsite_disorder(w, seed=10)
    print(system_tb.hamiltonian)

    print(np.allclose(system_py.hamiltonian, system_tb.hamiltonian, atol=1e-10))

if __name__=='__main__' :
    argh.dispatch_command(test_km)