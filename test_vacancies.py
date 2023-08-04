import numpy as np
from spinv import classes, finite_model, supercell
from spinv.example_models import kane_mele_tbmodels, kane_mele_pythtb, haldane_pythtb, haldane_tbmodels
from time import time
import scipy.linalg as la
import matplotlib.pyplot as plt

pbcmodel = haldane_tbmodels(0.3, 1, 0.15, np.pi / 2, 1)

# Make supercell and the relative finite model
super_model = supercell.Supercell(pbcmodel, 8, 8, False)
fin_model = finite_model.FiniteModel(pbcmodel, 8, 8, False)

# Make vacancies list (5 vacancies)
random_vacancies = [[np.random.randint(8), np.random.randint(8), np.random.randint(2)] for _ in range(20)]

# Add vacancies
super_model.add_vacancies(random_vacancies)
fin_model.add_vacancies(random_vacancies)

# Check that the same sites have been removed
assert np.allclose(super_model.r, fin_model.r)

# Single point calculation
print("Single point Chern number: ", super_model.single_point_chern("symmetric"))

# Local marker calculation
fig, ax = plt.subplots(1, 1)
ax.set_title("Single point Chern number: {0}".format(super_model.single_point_chern("symmetric")))
lcm = fin_model.local_chern_marker()
pos = fin_model.cart_positions
ax.scatter(pos[:, 0], pos[:, 1], c = lcm)
fig.savefig("out.pdf")

print("Produced plot")