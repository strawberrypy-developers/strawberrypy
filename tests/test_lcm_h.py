import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from strawberrypy import local_chern_marker, make_finite, onsite_disorder
from strawberrypy.example_models import haldane_pythtb, haldane_tbmodels

def test_lcm_haldane():
    # Construction of the Haldane model with TBmodels
    hmodel_tbm = haldane_tbmodels(delta = 0.5, t = -1, t2 = 0.15, phi = np.pi / 2, L = 1)
    hmodel_tbm = make_finite(model = hmodel_tbm, nx_sites = 10, ny_sites = 10)

    # Evaluation of the local Chern marker along the x (0) direction at fixed y = 5
    lcm_tbm = local_chern_marker(model = hmodel_tbm, nx_sites = 10, ny_sites = 10, direction = 0, start = 5)

    # Construction of the Haldane model with PythTB
    hmodel_pythtb = haldane_pythtb(delta = 0.5, t = -1, t2 = 0.15, phi = np.pi / 2, L = 1)
    hmodel_pythtb = make_finite(model = hmodel_pythtb, nx_sites = 10, ny_sites = 10)

    # Evaluation of the local Chern marker along the x (0) direction at fixed y = 5
    lcm_pythtb = local_chern_marker(model = hmodel_pythtb, nx_sites = 10, ny_sites = 10, direction = 0, start = 5)

    # Assert test for the two results
    assert np.allclose(lcm_pythtb, lcm_tbm)

    # Add disorder to the models
    hmodel_tbm_disorder = onsite_disorder(model = hmodel_tbm, w = 1, spinstates = 1, seed = 184)
    hmodel_pythtb_disorder = onsite_disorder(model = hmodel_pythtb, w = 1, spinstates = 1, seed = 184)

    # Evaluation of the local Chern marker along the x (0) direction at fixed y = 5, averaging over a region of radius 2
    lcm_tbm = local_chern_marker(model = hmodel_tbm_disorder, nx_sites = 10, ny_sites = 10, direction = 0, start = 5, macroscopic_average = True, cutoff = 2)
    lcm_pythtb = local_chern_marker(model = hmodel_pythtb_disorder, nx_sites = 10, ny_sites = 10, direction = 0, start = 5, macroscopic_average = True, cutoff = 2)

    # Assert test for the two results
    assert np.allclose(lcm_pythtb, lcm_tbm)