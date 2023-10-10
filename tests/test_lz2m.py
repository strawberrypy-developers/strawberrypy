import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from strawberrypy import FiniteModel
from strawberrypy.example_models import kane_mele_tbmodels, kane_mele_pythtb

def test_local_z2_marker_kane_mele():
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

    # Check equality for the local Z2 marker in the pristine case
    lz2m_ptb = kmmodel_ptb_finite.local_Z2_marker()
    lz2m_tbm = kmmodel_tbm_finite.local_Z2_marker()
    assert np.allclose(lz2m_ptb, lz2m_tbm)

    # Add onsite Anderson disorder
    kmmodel_ptb_finite.add_onsite_disorder(w = 3, seed = seed)
    kmmodel_tbm_finite.add_onsite_disorder(w = 3, seed = seed)

    # Check equality for the local Z2 marker in the disordered case
    lz2m_ptb = kmmodel_ptb_finite.local_Z2_marker(macroscopic_average = True, cutoff = 1.5)
    lz2m_tbm = kmmodel_tbm_finite.local_Z2_marker(macroscopic_average = True, cutoff = 1.5)
    assert np.allclose(lz2m_ptb, lz2m_tbm)