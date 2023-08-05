import numpy as np
import argh
import time
from strawberrypy.supercell import Supercell
from strawberrypy.example_models import kane_mele_pythtb, kane_mele_tbmodels, km_anderson_disorder_pythtb, km_anderson_disorder_tbmodels
from strawberrypy.example_models import haldane_pythtb, haldane_tbmodels, h_anderson_disorder_pythtb, h_anderson_disorder_tbmodels

def test_spscn(L=42, r=1., e=3., spin_o=0.3, w=2., spin_chern='down', which_formula='both'):
#inputs are:    linear size of supercell LxL
#               r = rashba/spin_orb
#               e = e_onsite/spin_orb
#               w = disorder stregth W/t
#               spin_chern = choice of spin Chern number  'up' or 'down' 
#               which_formula = choice of single point formula 'asymmetric', 'symmetric' or 'both'     

    #create Kane-Mele model in supercell LxL through PythTB package
    #km_pythtb_model = kane_mele_pythtb(r,e,spin_o,L=1)

    #create Kane-Mele model in supercell LxL through TBmodels package
    km_tbmodels_model = kane_mele_tbmodels(r,e,spin_o,L=1)

    km_sup = Supercell(km_tbmodels_model, spinful=True, Lx=L, Ly=L)

    print(km_sup.n_occ)

    np.random.seed(19)
    # Make vacancies list (5 vacancies)
    random_vacancies = [[np.random.randint(L), np.random.randint(L), np.random.randint(2)] for _ in range(2)]

    # Add vacancies
    km_sup.add_vacancies(vacancies_list = random_vacancies)

    print(km_sup.n_occ)

    spin_chern = km_sup.single_point_spin_chern(spin='down', formula='both', return_ham_gap=True, return_pszp_gap=True)

    print(spin_chern)

if __name__=='__main__' :
    argh.dispatch_command(test_spscn)