import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from strawberrypy import FiniteModel
from strawberrypy.example_models import kane_mele_tbmodels, kane_mele_pythtb, haldane_tbmodels, haldane_pythtb
import matplotlib.pyplot as plt

def test_heterostructure_haldane():
    # Test on the spinless Haldane model
    pbcmodel1 = haldane_tbmodels(delta = 0.3, t = 1, t2 = 0.15, phi = np.pi / 2, L = 1)
    pbcmodel2 = haldane_tbmodels(delta = 10.3, t = 1, t2 = 0.15, phi = np.pi / 2, L = 1)

    # Make the finite model
    fin_model = FiniteModel(tbmodel = pbcmodel1, Lx = 16, Ly = 8, spinful = False)
    fin_model2 = FiniteModel(tbmodel = pbcmodel2, Lx = 16, Ly = 8, spinful = False)
    fin_model.make_heterostructure(model2 = fin_model2, direction = 0, start = 0, stop = 7)

    ##  Example usage, Haldane model
    # fig, ax = plt.subplots(1, 1)
    # ax.set_title("Local Chern marker")
    # lcm = fin_model.local_chern_marker()
    # pos = fin_model.cart_positions
    # ax.scatter(pos[:, 0], pos[:, 1], c = lcm)
    # fig.savefig("localchernmarker.pdf")

def test_heterostructure_kane_mele():
    # Test on the spinful Kane-Mele model
    pbcmodel1 = kane_mele_tbmodels(rashba = 0.3, esite = 0.5, spin_orb = 0.3, L = 1)
    pbcmodel2 = kane_mele_tbmodels(rashba = 0.3, esite = 8.5, spin_orb = 0.3, L = 1)

    # Make the finite model
    fin_model = FiniteModel(tbmodel = pbcmodel1, Lx = 16, Ly = 8, spinful = True)
    fin_model2 = FiniteModel(tbmodel = pbcmodel2, Lx = 16, Ly = 8, spinful = True)
    fin_model.make_heterostructure(model2 = fin_model2, direction = 0, start = 0, stop = 7)

    ##  Example usage, Kane-Mele model
    # print("Single point spin Chern number: ", super_model.single_point_spin_chern(formula = "symmetric"))
    # fig, ax = plt.subplots(1, 1)
    # ax.set_title("Single point spin Chern number: {0}".format(super_model.single_point_spin_chern(formula = "symmetric")))
    # lscm = fin_model.local_spin_chern_marker()
    # pos = fin_model.cart_positions
    # ax.scatter(pos[:, 0], pos[:, 1], c = lscm)
    # fig.savefig("localspinchernmarker.pdf")