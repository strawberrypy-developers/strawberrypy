import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from strawberrypy import FiniteModel
from strawberrypy.example_models import haldane_pythtb, haldane_tbmodels, kane_mele_pythtb, kane_mele_tbmodels
from strawberrypy.utils import unique_vacancies

def test_local_chern_marker_haldane_tbmodels_pythtb():
    # Define the PBC models
    hmodel_ptb = haldane_pythtb(delta = 0.3, t = 1, t2 = 0.15, phi = np.pi / 2)
    hmodel_tbm = haldane_tbmodels(delta = 0.3, t = 1, t2 = 0.15, phi = np.pi / 2)

    # Build the finite model instance
    hmodel_ptb_finite = FiniteModel(tbmodel = hmodel_ptb, Lx = 8, Ly = 8, spinful = False)
    hmodel_tbm_finite = FiniteModel(tbmodel = hmodel_tbm, Lx = 8, Ly = 8, spinful = False)

    assert np.allclose(hmodel_ptb_finite.r, hmodel_tbm_finite.r)
    assert np.allclose(hmodel_ptb_finite.hamiltonian, hmodel_tbm_finite.hamiltonian)

    # Check equality for the local Chern marker in the pristine case
    lcm_ptb = hmodel_ptb_finite.local_chern_marker()
    lcm_tbm = hmodel_tbm_finite.local_chern_marker()
    assert np.allclose(lcm_ptb, lcm_tbm)

    # Set seed for disorder
    seed = 12345

    # Add onsite Anderson disorder
    hmodel_ptb_finite.add_onsite_disorder(w = 3, seed = seed)
    hmodel_tbm_finite.add_onsite_disorder(w = 3, seed = seed)

    # Check equality for the local Chern marker in the disordered case
    lcm_ptb = hmodel_ptb_finite.local_chern_marker(macroscopic_average = True, cutoff = 1.5)
    lcm_tbm = hmodel_tbm_finite.local_chern_marker(macroscopic_average = True, cutoff = 1.5)
    assert np.allclose(lcm_ptb, lcm_tbm)

def test_local_chern_marker_haldane_static():
    check_lcm = np.loadtxt("./tests/check/lcm_uniform_disorder.check")
    check_positions = np.loadtxt("./tests/check/positions_haldane_uniform.check")

    # Arbitrary order to check equality
    check_x, check_y, check_lcm = zip(*sorted(zip(check_positions[:, 0], check_positions[:, 1], check_lcm)))

    hmodel = haldane_tbmodels(delta = 0.3, t = 1, t2 = 0.15, phi = np.pi / 2)
    hmodel_finite = FiniteModel(tbmodel = hmodel, Lx = 8, Ly = 8, spinful = False)

    # Set seed for disorder
    seed = 12345
    hmodel_finite.add_onsite_disorder(w = 3, seed = seed)
    
    lcm = hmodel_finite.local_chern_marker(macroscopic_average = True, cutoff = 1.5)
    positions = hmodel_finite.cart_positions

    # Arbitrary order to check equality
    x, y, lcm = zip(*sorted(zip(positions[:, 0], positions[:, 1], lcm)))

    assert np.allclose(check_x, x)
    assert np.allclose(check_y, y)
    assert np.allclose(check_lcm, lcm)

def test_local_chern_marker_haldane_heterostructure_static():
    check_lcm = np.loadtxt("./tests/check/lcm_heterostructure.check")
    check_positions = np.loadtxt("./tests/check/positions_haldane_heterostructure.check")

    # Arbitrary order to check equality
    check_x, check_y, check_lcm = zip(*sorted(zip(check_positions[:, 0], check_positions[:, 1], check_lcm)))

    hmodel = haldane_tbmodels(delta = 0.3, t = 1, t2 = 0.15, phi = np.pi / 2)
    hmodel_finite = FiniteModel(tbmodel = hmodel, Lx = 16, Ly = 8, spinful = False)

    hmodel_2 = haldane_tbmodels(delta = 5.3, t = 1, t2 = 0.15, phi = np.pi / 2)
    hmodel_2_finite = FiniteModel(tbmodel = hmodel_2, Lx = 16, Ly = 8, spinful = False)
    
    # Build the heterostructure
    hmodel_finite.make_heterostructure(model2 = hmodel_2_finite, direction = 0, start = 0, stop = 7)
    
    lcm = hmodel_finite.local_chern_marker()
    positions = hmodel_finite.cart_positions

    # Arbitrary order to check equality
    x, y, lcm = zip(*sorted(zip(positions[:, 0], positions[:, 1], lcm)))

    assert np.allclose(check_x, x)
    assert np.allclose(check_y, y)
    assert np.allclose(check_lcm, lcm)

def test_local_chern_marker_haldane_vacancies_static():
    check_lcm = np.loadtxt("./tests/check/lcm_vacancies.check")
    check_positions = np.loadtxt("./tests/check/positions_vacancies.check")

    # Arbitrary order to check equality
    check_x, check_y, check_lcm = zip(*sorted(zip(check_positions[:, 0], check_positions[:, 1], check_lcm)))

    hmodel = haldane_tbmodels(delta = 0.3, t = 1, t2 = 0.15, phi = np.pi / 2)
    hmodel_finite = FiniteModel(tbmodel = hmodel, Lx = 8, Ly = 8, spinful = False)

    # Set seed for disorder
    seed = 12345
    vacancies = unique_vacancies(num = 10, Lx = 8, Ly = 8, basis = 2, seed = seed)
    hmodel_finite.add_vacancies(vacancies_list = vacancies)
    
    lcm = hmodel_finite.local_chern_marker(macroscopic_average = True, cutoff = 1.5)
    positions = hmodel_finite.cart_positions

    # Arbitrary order to check equality
    x, y, lcm = zip(*sorted(zip(positions[:, 0], positions[:, 1], lcm)))

    assert np.allclose(check_x, x)
    assert np.allclose(check_y, y)
    assert np.allclose(check_lcm, lcm)

def test_localization_marker_haldane_tbmodels_pythtb():
    # Define the PBC models
    hmodel_ptb = haldane_pythtb(delta = 0.3, t = 1, t2 = 0.15, phi = np.pi / 2)
    hmodel_tbm = haldane_tbmodels(delta = 0.3, t = 1, t2 = 0.15, phi = np.pi / 2)

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

def test_localization_marker_haldane_static():
    check_loc = np.loadtxt("./tests/check/loc_haldane_uniform_disorder.check")
    check_positions = np.loadtxt("./tests/check/positions_haldane_uniform.check")

    # Arbitrary order to check equality
    check_x, check_y, check_loc = zip(*sorted(zip(check_positions[:, 0], check_positions[:, 1], check_loc)))

    hmodel = haldane_tbmodels(delta = 0.3, t = 1, t2 = 0.15, phi = np.pi / 2)
    hmodel_finite = FiniteModel(tbmodel = hmodel, Lx = 8, Ly = 8, spinful = False)

    # Set seed for disorder
    seed = 12345
    hmodel_finite.add_onsite_disorder(w = 3, seed = seed)
    
    loc = hmodel_finite.localization_marker(macroscopic_average = True, cutoff = 1.5)
    positions = hmodel_finite.cart_positions

    # Arbitrary order to check equality
    x, y, loc = zip(*sorted(zip(positions[:, 0], positions[:, 1], loc)))

    assert np.allclose(check_x, x)
    assert np.allclose(check_y, y)
    assert np.allclose(check_loc, loc)

def test_localization_marker_haldane_heterostructure_static():
    check_loc = np.loadtxt("./tests/check/loc_haldane_heterostructure.check")
    check_positions = np.loadtxt("./tests/check/positions_haldane_heterostructure.check")

    # Arbitrary order to check equality
    check_x, check_y, check_loc = zip(*sorted(zip(check_positions[:, 0], check_positions[:, 1], check_loc)))

    hmodel = haldane_tbmodels(delta = 0.3, t = 1, t2 = 0.15, phi = np.pi / 2)
    hmodel_finite = FiniteModel(tbmodel = hmodel, Lx = 16, Ly = 8, spinful = False)

    hmodel_2 = haldane_tbmodels(delta = 5.3, t = 1, t2 = 0.15, phi = np.pi / 2)
    hmodel_2_finite = FiniteModel(tbmodel = hmodel_2, Lx = 16, Ly = 8, spinful = False)
    
    # Build the heterostructure
    hmodel_finite.make_heterostructure(model2 = hmodel_2_finite, direction = 0, start = 0, stop = 7)
    
    loc = hmodel_finite.localization_marker()
    positions = hmodel_finite.cart_positions

    # Arbitrary order to check equality
    x, y, loc = zip(*sorted(zip(positions[:, 0], positions[:, 1], loc)))

    assert np.allclose(check_x, x)
    assert np.allclose(check_y, y)
    assert np.allclose(check_loc, loc)

def test_localization_marker_kane_mele_tbmodels_pythtb():
    # Define the PBC models
    kmmodel_ptb = kane_mele_pythtb(rashba = 0.3, esite = 1, spin_orb = 0.1)
    kmmodel_tbm = kane_mele_tbmodels(rashba = 0.3, esite = 1, spin_orb = 0.1)

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

def test_localization_marker_kanemele_static():
    check_loc = np.loadtxt("./tests/check/loc_kanemele_uniform_disorder.check")
    check_positions = np.loadtxt("./tests/check/positions_kanemele_uniform.check")

    # Arbitrary order to check equality
    check_x, check_y, check_loc = zip(*sorted(zip(check_positions[:, 0], check_positions[:, 1], check_loc)))

    kmmodel = kane_mele_tbmodels(rashba = 0.3, esite = 1, spin_orb = 0.1)
    kmmodel_finite = FiniteModel(tbmodel = kmmodel, Lx = 8, Ly = 8, spinful = True)

    # Set seed for disorder
    seed = 12345
    kmmodel_finite.add_onsite_disorder(w = 3, seed = seed)
    
    loc = kmmodel_finite.localization_marker(macroscopic_average = True, cutoff = 1.5)
    positions = kmmodel_finite.cart_positions

    # Arbitrary order to check equality
    x, y, loc = zip(*sorted(zip(positions[:, 0], positions[:, 1], loc)))

    assert np.allclose(check_x, x)
    assert np.allclose(check_y, y)
    assert np.allclose(check_loc, loc)

def test_localization_marker_kanemele_heterostructure_static():
    check_loc = np.loadtxt("./tests/check/loc_kanemele_heterostructure.check")
    check_positions = np.loadtxt("./tests/check/positions_kanemele_heterostructure.check")

    # Arbitrary order to check equality
    check_x, check_y, check_loc = zip(*sorted(zip(check_positions[:, 0], check_positions[:, 1], check_loc)))

    kmmodel = kane_mele_tbmodels(rashba = 0.3, esite = 1, spin_orb = 0.1)
    kmmodel_finite = FiniteModel(tbmodel = kmmodel, Lx = 16, Ly = 8, spinful = True)

    kmmodel_2 = kane_mele_tbmodels(rashba = 0.3, esite = 5, spin_orb = 0.1)
    kmmodel_2_finite = FiniteModel(tbmodel = kmmodel_2, Lx = 16, Ly = 8, spinful = True)
    
    # Build the heterostructure
    kmmodel_finite.make_heterostructure(model2 = kmmodel_2_finite, direction = 0, start = 0, stop = 7)
    
    loc = kmmodel_finite.localization_marker()
    positions = kmmodel_finite.cart_positions

    # Arbitrary order to check equality
    x, y, loc = zip(*sorted(zip(positions[:, 0], positions[:, 1], loc)))

    assert np.allclose(check_x, x)
    assert np.allclose(check_y, y)
    assert np.allclose(check_loc, loc)