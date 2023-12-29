import numpy as np
import math

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from strawberrypy.supercell import Supercell
from strawberrypy.example_models import kane_mele_pythtb, kane_mele_tbmodels

def test_spscn(L=6, r=1., e=3., spin_o=0.3, w=2.):
#inputs are:    linear size of supercell LxL
#               r = rashba/spin_orb
#               e = e_onsite/spin_orb
#               w = disorder stregth W/t
#               spin_chern = choice of spin Chern number  'up' or 'down' 
#               which_formula = choice of single point formula 'asymmetric', 'symmetric' or 'both'  
      
    #create Kane-Mele model in the primitive cell through PythTB package
    km_pythtb = kane_mele_pythtb(r,e,spin_o)

    #create Kane-Mele model in the primitive cell through TBmodels package
    km_tbmodels = kane_mele_tbmodels(r,e,spin_o)

    #initialize supercell models
    system_pytb = Supercell(km_pythtb, Lx=L, Ly=L, spinful=True)
    system_tbm = Supercell(km_tbmodels, Lx=L, Ly=L, spinful=True)

    #add Anderson disorder
    system_tbm.add_onsite_disorder(w,seed=10)
    system_pytb.add_onsite_disorder(w,seed=10)

    # Single Point Spin Chern Number (SPSCN) calculation for models created with both packages, for the same disorder configuration

    spin_chern_pythtb, pszp_gap_pythtb, ham_gap_pythtb = system_pytb.single_point_spin_chern(formula='both', return_pszp_gap = True, return_ham_gap = True)
    spin_chern_tbmodels, pszp_gap_tbmodels, ham_gap_tbmodels = system_tbm.single_point_spin_chern(formula='both', return_pszp_gap = True, return_ham_gap = True)
    
    print('PythTB package, supercell size L =', L, ' disorder strength = ', w,  ' SPSCN :', spin_chern_pythtb['symmetric'] )
    print('TBmodels package, supercell size L =', L, ' disorder strength = ', w,  ' SPSCN :', spin_chern_tbmodels['symmetric'] )

    assert math.isclose(spin_chern_pythtb['asymmetric'],spin_chern_tbmodels['asymmetric'],abs_tol=1e-10)
    assert math.isclose(spin_chern_pythtb['symmetric'],spin_chern_tbmodels['symmetric'],abs_tol=1e-10)
    assert math.isclose(pszp_gap_pythtb,pszp_gap_tbmodels,abs_tol=1e-10)
    assert math.isclose(ham_gap_pythtb,ham_gap_tbmodels,abs_tol=1e-10)
