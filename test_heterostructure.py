import numpy as np
from spinv import classes, finite_model
from spinv.example_models import kane_mele_tbmodels, kane_mele_pythtb, haldane_pythtb, haldane_tbmodels
from time import time
import scipy.linalg as la
import strawberrypy
import matplotlib.pyplot as plt

model1 = haldane_tbmodels(0.3, 1, 0.15, np.pi / 2, 1)
model2 = haldane_tbmodels(1.25, 1, 0.15, np.pi / 2, 1)

model1finite = strawberrypy.make_finite(model1, 12, 6)
model2finite = strawberrypy.make_finite(model2, 12, 6)

heterostructure_old = strawberrypy.make_heterostructure(model1finite, model2finite, 12, 6, 0, 0, 5)

lcm_old = strawberrypy.local_chern_marker(heterostructure_old, 12, 6)

model1finite = finite_model.FiniteModel(model1, 12, 6, False)
model2finite = finite_model.FiniteModel(model2, 12, 6, False)
model1finite.make_heterostructure(model2finite, 0, 0, 5)
lcm_new = model1finite.local_chern_marker()

fig, ax = plt.subplots(1, 1)
pos = model1finite.cart_positions.T
ax.scatter(pos[0], pos[1], c = lcm_new)
fig.savefig("out.pdf")

assert np.allclose(lcm_old, lcm_new)
print("Test passed")