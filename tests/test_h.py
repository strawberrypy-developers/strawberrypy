import numpy as np
import math

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from strawberrypy import single_point_chern

from strawberrypy.example_models import haldane_pythtb, haldane_tbmodels, h_anderson_disorder_pythtb, h_anderson_disorder_tbmodels

def test_spcn(L=6, t=-4., t2=1., delta=2., pi_phi=-2., w=1.5, which_formula = 'symmetric'):
#inputs are:    linear size of supercell LxL
#               t = first neighbours real hopping
#               t2 = second neighbours
#               delta = energy on site
#               pi_phi --> phi = pi/(pi_phi)  where phi = second neighbours hopping phase
#               w = disorder stregth W/t
#               which_formula = choice of single point formula 'asymmetric', 'symmetric' or 'both'     

    #Haldane model parameters
    phi = np.pi/pi_phi

    #create Haldane model in supercell LxL through PythTB package
    h_pythtb_model = haldane_pythtb(delta, t, t2, phi, L)

    #create Haldane model in supercell LxL through TBmodels package
    h_tbmodels_model = haldane_tbmodels(delta, t, t2, phi, L)

    #add Anderson disorder in PythTB model
    np.random.seed(15)
    h_pythtb_model = h_anderson_disorder_pythtb(h_pythtb_model, w)

    #add Anderson disorder in TBmodels model
    np.random.seed(15)
    h_tbmodels_model = h_anderson_disorder_tbmodels(h_tbmodels_model, w)

    # Single Point Chern Number (SPCN) calculation for models created with both packages, for the same disorder configuration

    chern_pythtb = single_point_chern(h_pythtb_model, formula=which_formula)

    chern_tbmodels = single_point_chern(h_tbmodels_model, formula=which_formula)

    # if which_formula = 'both', then Single Point Chern Numbers are printed as follows : 'asymmetric' 'symmetric'
    print('PythTB package, supercell size L =', L, ' disorder strength = ', w,  ' SPCN :', *chern_pythtb )
    print('TBmodels package, supercell size L =', L, ' disorder strength = ', w,  ' SPCN :', *chern_tbmodels )

    assert math.isclose(chern_pythtb[0],chern_tbmodels[0],abs_tol=1e-10)