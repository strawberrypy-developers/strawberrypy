import numpy as np
import math

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from strawberrypy import single_point_spin_chern

from strawberrypy.example_models import kane_mele_pythtb, kane_mele_tbmodels, km_anderson_disorder_pythtb, km_anderson_disorder_tbmodels

def test_spscn(L=6, r=1., e=3., spin_o=0.3, w=2., spin_chern='down', which_formula='symmetric'):
#inputs are:    linear size of supercell LxL
#               r = rashba/spin_orb
#               e = e_onsite/spin_orb
#               w = disorder stregth W/t
#               spin_chern = choice of spin Chern number  'up' or 'down' 
#               which_formula = choice of single point formula 'asymmetric', 'symmetric' or 'both'        

    #create Kane-Mele model in supercell LxL through PythTB package
    km_pythtb_model = kane_mele_pythtb(r,e,spin_o,L)

    #create Kane-Mele model in supercell LxL through TBmodels package
    km_tbmodels_model = kane_mele_tbmodels(r,e,spin_o,L)

    #add Anderson disorder in PythTB model
    np.random.seed(15)
    km_pythtb_model = km_anderson_disorder_pythtb(km_pythtb_model,w)

    #add Anderson disorder in TBmodels model
    np.random.seed(15)
    km_tbmodels_model = km_anderson_disorder_tbmodels(km_tbmodels_model, w)

    # Single Point Spin Chern Number (SPSCN) calculation for models created with both packages, for the same disorder configuration

    spin_chern_pythtb = single_point_spin_chern(km_pythtb_model, spin=spin_chern, formula=which_formula)

    spin_chern_tbmodels = single_point_spin_chern(km_tbmodels_model, spin=spin_chern, formula=which_formula)

    # if which_formula = 'both', then Single Point Spin Chern numbers are printed as follows : 'asymmetric' 'symmetric'
    print('PythTB package, supercell size L =', L, ' disorder strength = ', w,  ' SPSCN :', *spin_chern_pythtb )
    print('TBmodels package, supercell size L =', L, ' disorder strength = ', w,  ' SPSCN :', *spin_chern_tbmodels )

    assert math.isclose(spin_chern_pythtb[0],spin_chern_tbmodels[0],abs_tol=1e-10)
  
