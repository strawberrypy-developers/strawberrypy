import numpy as np
from spinv import classes, finite_model
from spinv.example_models import kane_mele_tbmodels, kane_mele_pythtb, haldane_pythtb, haldane_tbmodels
from time import time
import scipy.linalg as la
import strawberrypy

hmodel_ptb = haldane_pythtb(0.3, 1, 0.15, np.pi / 2, 1)
hmodel_tbm = haldane_tbmodels(0.3, 1, 0.15, np.pi / 2, 1)

seed = 1982458

# Old strawberrypy test - Beware: some changes in the averages -> required update to pass the tests
t = time()
hmodel_ptb_finite = strawberrypy.make_finite(hmodel_ptb, 8, 8)
hmodel_tbm_finite = strawberrypy.make_finite(hmodel_tbm, 8, 8)

lcm_ptb = strawberrypy.local_chern_marker(hmodel_ptb_finite, 8, 8)
lcm_tbm = strawberrypy.local_chern_marker(hmodel_tbm_finite, 8, 8)
#print("Old strawberrypy: ", time() - t)

assert np.allclose(lcm_ptb, lcm_tbm)

# New version
t = time()
hmodel_ptb_finite_class = finite_model.FiniteModel(hmodel_ptb, 8, 8, False)
hmodel_tbm_finite_class = finite_model.FiniteModel(hmodel_tbm, 8, 8, False)

assert np.allclose(hmodel_ptb_finite_class.r, hmodel_tbm_finite_class.r)
assert np.allclose(hmodel_ptb_finite_class.hamiltonian, hmodel_tbm_finite_class.hamiltonian)

lcm_ptb_2 = hmodel_ptb_finite_class.local_chern_marker()
lcm_tbm_2 = hmodel_tbm_finite_class.local_chern_marker()
#print("New strawberrypy: ", time() - t)

assert np.allclose(lcm_ptb_2, lcm_tbm_2)

# Final check
assert np.allclose(lcm_ptb_2, lcm_tbm)

# Disorder - old method
hmodel_ptb_finite = strawberrypy.onsite_disorder(hmodel_ptb_finite, 3, 1, seed)
hmodel_tbm_finite = strawberrypy.onsite_disorder(hmodel_tbm_finite, 3, 1, seed)
lcm_ptb = strawberrypy.local_chern_marker(hmodel_ptb_finite, 8, 8, macroscopic_average = True, cutoff = 1.5)
lcm_tbm = strawberrypy.local_chern_marker(hmodel_tbm_finite, 8, 8, macroscopic_average = True, cutoff = 1.5)
assert np.allclose(lcm_ptb, lcm_tbm)

# Disorder - new method
hmodel_ptb_finite_class.add_onsite_disorder(3, seed)
hmodel_tbm_finite_class.add_onsite_disorder(3, seed)
lcm_ptb_2 = hmodel_ptb_finite_class.local_chern_marker(macroscopic_average = True, cutoff = 1.5)
lcm_tbm_2 = hmodel_tbm_finite_class.local_chern_marker(macroscopic_average = True, cutoff = 1.5)
assert np.allclose(lcm_ptb_2, lcm_tbm_2)

# Final check with disorder
assert np.allclose(lcm_ptb_2, lcm_tbm)


print("Test passed")