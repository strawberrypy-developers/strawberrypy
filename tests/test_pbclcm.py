import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from strawberrypy import Supercell
from strawberrypy.example_models import haldane_pythtb, haldane_tbmodels

def test_PBC_local_chern_marker_haldane():
    # Define the PBC models
    hmodel_ptb = haldane_pythtb(delta = 0.3, t = 1, t2 = 0.15, phi = np.pi / 2, L = 1)
    hmodel_tbm = haldane_tbmodels(delta = 0.3, t = 1, t2 = 0.15, phi = np.pi / 2, L = 1)

    # Set seed for disorder
    seed = 1982458

    # Build the supercell model instance
    hmodel_ptb_supercell = Supercell(tbmodel = hmodel_ptb, Lx = 8, Ly = 8, spinful = False)
    hmodel_tbm_supercell = Supercell(tbmodel = hmodel_tbm, Lx = 8, Ly = 8, spinful = False)

    assert np.allclose(hmodel_ptb_supercell.r, hmodel_tbm_supercell.r)
    assert np.allclose(hmodel_ptb_supercell.hamiltonian, hmodel_tbm_supercell.hamiltonian)

    # Check equality for the PBC local Chern marker in the pristine case
    lcm_ptb = hmodel_ptb_supercell.pbc_local_chern_marker()
    lcm_tbm = hmodel_tbm_supercell.pbc_local_chern_marker()
    assert np.allclose(lcm_tbm, lcm_ptb)

    # Add onsite Anderson disorder
    hmodel_ptb_supercell.add_onsite_disorder(w = 3, seed = seed)
    hmodel_tbm_supercell.add_onsite_disorder(w = 3, seed = seed)

    # Check equality for the PBC local Chern marker in the disordered case
    lcm_ptb = hmodel_ptb_supercell.pbc_local_chern_marker(macroscopic_average = True, cutoff = 1.5)
    lcm_tbm = hmodel_tbm_supercell.pbc_local_chern_marker(macroscopic_average = True, cutoff = 1.5)
    assert np.allclose(lcm_ptb, lcm_tbm)