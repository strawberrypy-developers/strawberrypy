import numpy as np

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from strawberrypy import FiniteModel
from strawberrypy.example_models import haldane_tbmodels, haldane_pythtb

def test_local_chern_marker():
    # Construction of the Haldane model with TBmodels and PythTB
    hmodel_tbm = haldane_tbmodels(delta = 0.5, t = -1, t2 = 0.15, phi = np.pi / 2, L = 1)
    hmodel_pythtb = haldane_pythtb(delta = 0.5, t = -1, t2 = 0.15, phi = np.pi / 2, L = 1)

    # Construct the two finite models
    hmodel_tbm = FiniteModel(tbmodel = hmodel_tbm, nx_sites = 10, ny_sites = 10, spinful = False, mode = 'tb')
    hmodel_pythtb = FiniteModel(tbmodel = hmodel_pythtb, nx_sites = 10, ny_sites = 10, spinful = False, mode = 'tb')

    # Compute the local Chern marker on the lattice
    lcm_tbm = hmodel_tbm.local_chern_marker()
    lcm_pythtb = hmodel_pythtb.local_chern_marker()

    # Check the two models give the same results
    assert np.allclose(lcm_tbm, lcm_pythtb)

    # Add Anderson disorder
    seed = 148364692
    hmodel_tbm.add_onsite_disorder(w = 0, seed = seed)
    hmodel_pythtb.add_onsite_disorder(w = 0, seed = seed)

    # Add random vacancies to the system by removing a lattice site
    random_vacancies = [[np.random.randint(10), np.random.randint(10), np.random.randint(2)] for _ in range(5)]
    hmodel_tbm.add_vacancies(vacancies_list = random_vacancies)
    hmodel_pythtb.add_vacancies(vacancies_list = random_vacancies)

    # Evaluate the local Chern marker averaging over a small region of the sample
    lcm_tbm = hmodel_tbm.local_chern_marker(direction = 0, start = 5, macroscopic_average = True, cutoff = 2)
    lcm_pythtb = hmodel_pythtb.local_chern_marker(direction = 0, start = 5, macroscopic_average = True, cutoff = 2)

    # Check the two models give the same results
    assert np.allclose(lcm_tbm, lcm_pythtb)