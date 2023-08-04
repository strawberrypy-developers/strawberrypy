import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from strawberrypy import FiniteModel
from strawberrypy.example_models import haldane_pythtb, haldane_tbmodels

def test_local_chern_marker_haldane():
    # Define the PBC models
    hmodel_ptb = haldane_pythtb(delta = 0.3, t = 1, t2 = 0.15, phi = np.pi / 2, L = 1)
    hmodel_tbm = haldane_tbmodels(delta = 0.3, t = 1, t2 = 0.15, phi = np.pi / 2, L = 1)

    # Set seed for disorder
    seed = 1982458

    # Build the finite model instance
    hmodel_ptb_finite = FiniteModel(tbmodel = hmodel_ptb, Lx = 8, Ly = 8, spinful = False)
    hmodel_tbm_finite = FiniteModel(tbmodel = hmodel_tbm, Lx = 8, Ly = 8, spinful = False)

    assert np.allclose(hmodel_ptb_finite.r, hmodel_tbm_finite.r)
    assert np.allclose(hmodel_ptb_finite.hamiltonian, hmodel_tbm_finite.hamiltonian)

    # Check equality for the local Chern marker in the pristine case
    lcm_ptb = hmodel_ptb_finite.local_chern_marker()
    lcm_tbm = hmodel_tbm_finite.local_chern_marker()
    assert np.allclose(lcm_ptb, lcm_tbm)

    # Add onsite Anderson disorder
    hmodel_ptb_finite.add_onsite_disorder(w = 3, seed = seed)
    hmodel_tbm_finite.add_onsite_disorder(w = 3, seed = seed)

    # Check equality for the local Chern marker in the disordered case
    lcm_ptb = hmodel_ptb_finite.local_chern_marker(macroscopic_average = True, cutoff = 1.5)
    lcm_tbm = hmodel_tbm_finite.local_chern_marker(macroscopic_average = True, cutoff = 1.5)
    assert np.allclose(lcm_ptb, lcm_tbm)