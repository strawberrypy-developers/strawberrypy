import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from strawberrypy import FiniteModel
from strawberrypy.example_models import kane_mele_pythtb, kane_mele_tbmodels, haldane_pythtb, haldane_tbmodels

def test_localization_marker_haldane():
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

    # Check equality for the localization marker in the pristine case
    lm_ptb = hmodel_ptb_finite.localization_marker()
    lm_tbm = hmodel_tbm_finite.localization_marker()
    assert np.allclose(lm_ptb, lm_tbm)

    # Add onsite Anderson disorder
    hmodel_ptb_finite.add_onsite_disorder(w = 3, seed = seed)
    hmodel_tbm_finite.add_onsite_disorder(w = 3, seed = seed)

    # Check equality for the localization marker in the disordered case
    lm_ptb = hmodel_ptb_finite.localization_marker(macroscopic_average = True, cutoff = 1.5)
    lm_tbm = hmodel_tbm_finite.localization_marker(macroscopic_average = True, cutoff = 1.5)
    assert np.allclose(lm_ptb, lm_tbm)

def test_localization_marker_kane_mele():
    # Define the PBC models
    kmmodel_ptb = kane_mele_pythtb(rashba = 0.3, esite = 1, spin_orb = 0.1, L = 1)
    kmmodel_tbm = kane_mele_tbmodels(rashba = 0.3, esite = 1, spin_orb = 0.1, L = 1)

    # Set seed for disorder
    seed = 1982458

    # Build the finite model instance
    kmmodel_ptb_finite = FiniteModel(tbmodel = kmmodel_ptb, Lx = 8, Ly = 8, spinful = True)
    kmmodel_tbm_finite = FiniteModel(tbmodel = kmmodel_tbm, Lx = 8, Ly = 8, spinful = True)

    assert np.allclose(kmmodel_ptb_finite.r, kmmodel_tbm_finite.r)
    assert np.allclose(kmmodel_ptb_finite.hamiltonian, kmmodel_tbm_finite.hamiltonian)

    # Check equality for the localization marker in the pristine case
    lm_ptb = kmmodel_ptb_finite.localization_marker()
    lm_tbm = kmmodel_tbm_finite.localization_marker()
    assert np.allclose(lm_ptb, lm_tbm)

    # Add onsite Anderson disorder
    kmmodel_ptb_finite.add_onsite_disorder(w = 3, seed = seed)
    kmmodel_tbm_finite.add_onsite_disorder(w = 3, seed = seed)

    # Check equality for the localization marker in the disordered case
    lm_ptb = kmmodel_ptb_finite.localization_marker(macroscopic_average = True, cutoff = 1.5)
    lm_tbm = kmmodel_tbm_finite.localization_marker(macroscopic_average = True, cutoff = 1.5)
    assert np.allclose(lm_ptb, lm_tbm)