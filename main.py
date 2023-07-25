import numpy as np
import argh
import time
from strawberrypy import single_point_spin_chern, single_point_chern
from strawberrypy._pythtb.lattice import orb_cart_spin, orb_cart
from strawberrypy.example_models import kane_mele_pythtb, kane_mele_tbmodels, km_anderson_disorder_pythtb, km_anderson_disorder_tbmodels
from strawberrypy.example_models import haldane_pythtb, haldane_tbmodels, h_anderson_disorder_pythtb, h_anderson_disorder_tbmodels

def test_spscn(L=24, r=1., e=3., spin_o=0.3, w=2., spin_chern='down', which_formula='both'):
#inputs are:    linear size of supercell LxL
#               r = rashba/spin_orb
#               e = e_onsite/spin_orb
#               w = disorder stregth W/t
#               spin_chern = choice of spin Chern number  'up' or 'down' 
#               which_formula = choice of single point formula 'asymmetric', 'symmetric' or 'both'     
    t1 = time.time()

    #create Kane-Mele model in supercell LxL through PythTB package
    km_pythtb_model = kane_mele_pythtb(r,e,spin_o,L)

    #create Kane-Mele model in supercell LxL through TBmodels package
    km_tbmodels_model = kane_mele_tbmodels(r,e,spin_o,L)

    #add Anderson disorder in PythTB model
    np.random.seed(10)
    km_pythtb_model = km_anderson_disorder_pythtb(km_pythtb_model,w)

    #add Anderson disorder in TBmodels model
    np.random.seed(10)
    km_tbmodels_model = km_anderson_disorder_tbmodels(km_tbmodels_model, w)

    # Single Point Spin Chern Number (SPSCN) calculation for models created with both packages, for the same disorder configuration

    spin_chern_pythtb = single_point_spin_chern(km_pythtb_model, spin=spin_chern, formula=which_formula)

    spin_chern_tbmodels = single_point_spin_chern(km_tbmodels_model, spin=spin_chern, formula=which_formula)

    # if which_formula = 'both', then Single Point Spin Chern numbers are printed as follows : 'asymmetric' 'symmetric'
    print('TBmodels package, supercell size L =', L, ' disorder strength = ', w,  ' SPSCN :', *spin_chern_tbmodels )
    print('PythTB package, supercell size L =', L, ' disorder strength = ', w,  ' SPSCN :', *spin_chern_pythtb )

    t2 = time.time()

    print(t2-t1)
    exit()

    point=[0.,0.]

    #eig,vec = km_pythtb_model.solve_one(point, eig_vectors=True)



    print(km_tbmodels_model.hamilton(point))


    exit()

    # Single Point Spin Chern Number (SPSCN) calculation for models created with both packages, for the same disorder configuration

    spin_chern_pythtb = single_point_spin_chern(km_pythtb_model, spin=spin_chern, formula=which_formula)

    spin_chern_tbmodels = single_point_spin_chern(km_tbmodels_model, spin=spin_chern, formula=which_formula)

    # if which_formula = 'both', then Single Point Spin Chern numbers are printed as follows : 'asymmetric' 'symmetric'
    print('PythTB package, supercell size L =', L, ' disorder strength = ', w,  ' SPSCN :', *spin_chern_pythtb )
    print('TBmodels package, supercell size L =', L, ' disorder strength = ', w,  ' SPSCN :', *spin_chern_tbmodels )


def test_spcn(L=3, t=-4., t2=1., delta=2., pi_phi=-2., w=1.5, which_formula = 'both'):
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


    # Single Point Chern Number (SPCN) calculation for models created with both packages, for the same disorder configuration

    chern_pythtb = single_point_chern(h_pythtb_model, formula=which_formula)

    chern_tbmodels = single_point_chern(h_tbmodels_model, formula=which_formula)

    # if which_formula = 'both', then Single Point Chern Numbers are printed as follows : 'asymmetric' 'symmetric'
    print('PythTB package, supercell size L =', L, ' disorder strength = ', w,  ' SPCN :', *chern_pythtb )
    print('TBmodels package, supercell size L =', L, ' disorder strength = ', w,  ' SPCN :', *chern_tbmodels )
    
    exit()

    #add Anderson disorder in PythTB model
    np.random.seed(15)
    h_pythtb_model = h_anderson_disorder_pythtb(h_pythtb_model, w)

    #add Anderson disorder in TBmodels model
    np.random.seed(10)
    h_tbmodels_model = h_anderson_disorder_tbmodels(h_tbmodels_model, w)

    # Single Point Chern Number (SPCN) calculation for models created with both packages, for the same disorder configuration

    chern_pythtb = single_point_chern(h_pythtb_model, formula=which_formula)

    chern_tbmodels = single_point_chern(h_tbmodels_model, formula=which_formula)

    # if which_formula = 'both', then Single Point Chern Numbers are printed as follows : 'asymmetric' 'symmetric'
    print('PythTB package, supercell size L =', L, ' disorder strength = ', w,  ' SPCN :', *chern_pythtb )
    print('TBmodels package, supercell size L =', L, ' disorder strength = ', w,  ' SPCN :', *chern_tbmodels )


if __name__=='__main__' :
    argh.dispatch_command(test_spscn)