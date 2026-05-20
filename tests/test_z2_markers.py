import numpy as np
import pytest

from strawberrypy import Model
from strawberrypy.example_models import kane_mele_tbmodels


@pytest.mark.mpi_skip()
def test_local_spin_chern_marker_kanemele_static():
    check_lscm = np.loadtxt("./check/lscm_uniform_disorder.check")
    check_positions = np.loadtxt("./check/positions_kanemele_uniform.check")

    # Arbitrary order to check equality
    check_x, check_y, check_lscm = zip(
        *sorted(zip(check_positions[:, 0], check_positions[:, 1], check_lscm))
    )

    kmmodel = kane_mele_tbmodels(rashba=1.0, esite=3.5, spin_orb=0.3)
    kmmodel_finite = Model(model=kmmodel, spinful=True).make_finite(Lx=8, Ly=8)

    # Set seed for disorder
    seed = 10
    kmmodel_finite.add_disorder_uniform(w=3, seed=seed)

    lscm = kmmodel_finite.local_spin_chern_marker(macroscopic_average=True, cutoff=1.5)
    positions = kmmodel_finite.cart_positions

    # Arbitrary order to check equality
    x, y, lscm = zip(*sorted(zip(positions[:, 0], positions[:, 1], lscm)))

    assert np.allclose(check_x, x)
    assert np.allclose(check_y, y)
    assert np.allclose(check_lscm, lscm)


@pytest.mark.mpi_skip()
def test_pbc_local_spin_chern_marker_kanemele_static():
    check_pbclscm = np.loadtxt("./check/pbclscm_uniform_disorder.check")
    check_positions = np.loadtxt("./check/positions_kanemele_uniform.check")

    # Arbitrary order to check equality
    check_x, check_y, check_pbclscm = zip(
        *sorted(zip(check_positions[:, 0], check_positions[:, 1], check_pbclscm))
    )

    kmmodel = kane_mele_tbmodels(rashba=1.0, esite=3.5, spin_orb=0.3)
    kmmodel_supercell = Model(model=kmmodel, spinful=True).make_supercell(Lx=8, Ly=8)

    # Set seed for disorder
    seed = 10
    kmmodel_supercell.add_disorder_uniform(w=3, seed=seed)

    pbclscm = kmmodel_supercell.pbc_local_spin_chern_marker(
        macroscopic_average=True, cutoff=1.5
    )
    positions = kmmodel_supercell.cart_positions

    # Arbitrary order to check equality
    x, y, pbclscm = zip(*sorted(zip(positions[:, 0], positions[:, 1], pbclscm)))

    assert np.allclose(check_x, x)
    assert np.allclose(check_y, y)
    assert np.allclose(check_pbclscm, pbclscm)


@pytest.mark.mpi_skip()
def test_local_z2_marker_kanemele_static():
    check_lz2m = np.loadtxt("./check/lz2m_uniform_disorder.check")
    check_positions = np.loadtxt("./check/positions_kanemele_uniform.check")

    # Arbitrary order to check equality
    check_x, check_y, check_lz2m = zip(
        *sorted(zip(check_positions[:, 0], check_positions[:, 1], check_lz2m))
    )

    kmmodel = kane_mele_tbmodels(rashba=1.0, esite=3.5, spin_orb=0.3)
    kmmodel_finite = Model(model=kmmodel, spinful=True).make_finite(Lx=8, Ly=8)

    # Set seed for disorder
    seed = 10
    kmmodel_finite.add_disorder_uniform(w=3, seed=seed)

    lz2m = kmmodel_finite.local_z2_marker(macroscopic_average=True, cutoff=1.5)
    positions = kmmodel_finite.cart_positions

    # Arbitrary order to check equality
    x, y, lz2m = zip(*sorted(zip(positions[:, 0], positions[:, 1], lz2m)))

    assert np.allclose(check_x, x)
    assert np.allclose(check_y, y)
    assert np.allclose(check_lz2m, lz2m)


@pytest.mark.mpi_skip()
def test_pbc_local_z2_marker_kanemele_static():
    check_pbclz2m = np.loadtxt("./check/pbclz2m_uniform_disorder.check")
    check_positions = np.loadtxt("./check/positions_kanemele_uniform.check")

    # Arbitrary order to check equality
    check_x, check_y, check_pbclz2m = zip(
        *sorted(zip(check_positions[:, 0], check_positions[:, 1], check_pbclz2m))
    )

    kmmodel = kane_mele_tbmodels(rashba=1.0, esite=3.5, spin_orb=0.3)
    kmmodel_supercell = Model(model=kmmodel, spinful=True).make_supercell(Lx=8, Ly=8)

    # Set seed for disorder
    seed = 10
    kmmodel_supercell.add_disorder_uniform(w=3, seed=seed)

    pbclz2m = kmmodel_supercell.pbc_local_z2_marker(macroscopic_average=True, cutoff=1.5)
    positions = kmmodel_supercell.cart_positions

    # Arbitrary order to check equality
    x, y, pbclz2m = zip(*sorted(zip(positions[:, 0], positions[:, 1], pbclz2m)))

    assert np.allclose(check_x, x)
    assert np.allclose(check_y, y)
    assert np.allclose(check_pbclz2m, pbclz2m)
