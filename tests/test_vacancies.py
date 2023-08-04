import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from strawberrypy import Supercell, FiniteModel
from strawberrypy.example_models import kane_mele_tbmodels, kane_mele_pythtb, haldane_pythtb, haldane_tbmodels

def test_vacancies_haldane():
    # Test on the spinless Haldane model
    pbcmodel = haldane_tbmodels(delta = 0.3, t = 1, t2 = 0.15, phi = np.pi / 2, L = 1)

    # Make supercell and the relative finite model
    super_model = Supercell(tbmodel = pbcmodel, Lx = 8, Ly = 8, spinful = False)
    fin_model = FiniteModel(tbmodel = pbcmodel, Lx = 8, Ly = 8, spinful = False)

    # Make vacancies list (5 vacancies)
    random_vacancies = [[np.random.randint(8), np.random.randint(8), np.random.randint(2)] for _ in range(20)]

    # Add vacancies
    super_model.add_vacancies(vacancies_list = random_vacancies)
    fin_model.add_vacancies(vacancies_list = random_vacancies)

    # Check that the same sites have been removed
    assert np.allclose(super_model.r, fin_model.r)
    assert np.allclose(super_model.cart_positions, fin_model.cart_positions)
    assert np.allclose(super_model.n_occ, fin_model.n_occ)
    assert np.allclose(super_model.n_orb, fin_model.n_orb)

    ##  Example usage, Haldane model
    # print("Single point Chern number: ", super_model.single_point_chern("symmetric"))
    # fig, ax = plt.subplots(1, 1)
    # ax.set_title("Single point Chern number: {0}".format(super_model.single_point_chern("symmetric")))
    # lcm = fin_model.local_chern_marker()
    # pos = fin_model.cart_positions
    # ax.scatter(pos[:, 0], pos[:, 1], c = lcm)
    # fig.savefig("localchernmarker.pdf")

def test_vacancies_kane_mele():
    # Test on the spinful Kane-Mele model
    pbcmodel = kane_mele_tbmodels(rashba = 0.3, esite = 0.5, spin_orb = 0.3, L = 1)

    # Make supercell and the relative finite model
    super_model = Supercell(tbmodel = pbcmodel, Lx = 8, Ly = 8, spinful = True)
    fin_model = FiniteModel(tbmodel = pbcmodel, Lx = 8, Ly = 8, spinful = True)

    # Make vacancies list (5 vacancies)
    random_vacancies = [[np.random.randint(8), np.random.randint(8), np.random.randint(2)] for _ in range(20)]

    # Add vacancies
    super_model.add_vacancies(vacancies_list = random_vacancies)
    fin_model.add_vacancies(vacancies_list = random_vacancies)

    # Check that the same sites have been removed
    assert np.allclose(super_model.r, fin_model.r)
    assert np.allclose(super_model.cart_positions, fin_model.cart_positions)
    assert np.allclose(super_model.sz, fin_model.sz)
    assert np.allclose(super_model.n_occ, fin_model.n_occ)
    assert np.allclose(super_model.n_orb, fin_model.n_orb)

    ##  Example usage, Kane-Mele model
    # print("Single point spin Chern number: ", super_model.single_point_spin_chern(formula = "symmetric"))
    # fig, ax = plt.subplots(1, 1)
    # ax.set_title("Single point spin Chern number: {0}".format(super_model.single_point_spin_chern(formula = "symmetric")))
    # lscm = fin_model.local_spin_chern_marker()
    # pos = fin_model.cart_positions
    # ax.scatter(pos[:, 0], pos[:, 1], c = lscm)
    # fig.savefig("localspinchernmarker.pdf")