import numpy as np
import math

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from strawberrypy.supercell import Supercell

from strawberrypy.example_models import haldane_pythtb, haldane_tbmodels

def test_spcn(L=6, t=-4., t2=1., delta=2., pi_phi=-2., w=1.5):
#inputs are:    linear size of supercell LxL
#               t = first neighbours real hopping
#               t2 = second neighbours
#               delta = energy on site
#               pi_phi --> phi = pi/(pi_phi)  where phi = second neighbours hopping phase
#               w = disorder stregth W/t
#               which_formula = choice of single point formula 'asymmetric', 'symmetric' or 'both'     

    #Haldane model parameters
    phi = np.pi/pi_phi

    #create Haldane model in the primitive cell through PythTB package
    h_pythtb_model = haldane_pythtb(delta,t,t2,phi)

    #create Haldane model in the primitive cell  through TBmodels package
    h_tbmodels_model = haldane_tbmodels(delta,t,t2,phi)

    #initialize supercell models 
    system_tbm = Supercell(h_tbmodels_model, Lx=L, Ly=L, spinful=False)
    system_pytb = Supercell(h_pythtb_model, Lx=L, Ly=L, spinful=False)

    #add Anderson disorder 
    system_pytb.add_onsite_disorder(w, seed=10)
    system_tbm.add_onsite_disorder(w, seed=10)

    # Single Point Chern Number (SPCN) calculation for models created with both packages, for the same disorder configuration

    chern_pythtb, ham_gap_pythtb = system_pytb.single_point_chern(formula='both', return_ham_gap=True)
    chern_tbmodels, ham_gap_tbmodels = system_tbm.single_point_chern(formula='both', return_ham_gap=True)

    print('PythTB package, supercell size L =', L, ' disorder strength = ', w,  ' SPCN :', chern_pythtb['symmetric'] )
    print('TBmodels package, supercell size L =', L, ' disorder strength = ', w,  ' SPCN :', chern_tbmodels['symmetric'] )

    assert math.isclose(chern_pythtb['asymmetric'],chern_tbmodels['asymmetric'],abs_tol=1e-10)
    assert math.isclose(chern_pythtb['symmetric'],chern_tbmodels['symmetric'],abs_tol=1e-10)
    assert math.isclose(ham_gap_pythtb,ham_gap_tbmodels,abs_tol=1e-10)