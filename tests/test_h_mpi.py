import numpy as np, math
import pytest

from strawberrypy import Model, example_models
from strawberrypy.config import DEBUG_MODE


@pytest.mark.mpi
def test_spcn(L=6, t=-4.0, t2=1.0, delta=2.0, pi_phi=-2.0, w=1.5):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # inputs are:   linear size of supercell LxL
    #               t = first neighbours real hopping
    #               t2 = second neighbours
    #               delta = energy on site
    #               pi_phi --> phi = pi/(pi_phi)  where phi = second neighbours hopping phase
    #               w = disorder stregth W/t
    #               which_formula = choice of single point formula 'asymmetric', 'symmetric' or 'both'

    # Haldane model parameters
    phi = np.pi / pi_phi

    # create Haldane model in the primitive cell through PythTB package
    h_pythtb_model = example_models.haldane_pythtb(delta, t, t2, phi)

    # create Haldane model in the primitive cell  through TBmodels package
    h_tbmodels_model = example_models.haldane_tbmodels(delta, t, t2, phi)

    # initialize supercell models
    system_tbm = Model(h_tbmodels_model, spinful=False).make_supercell(Lx=L, Ly=L)
    system_pytb = Model(h_pythtb_model, spinful=False).make_supercell(Lx=L, Ly=L)

    # add Anderson disorder
    system_pytb.add_disorder_uniform(w, seed=10)
    system_tbm.add_disorder_uniform(w, seed=10)

    # Single Point Chern Number (SPCN) calculation for models created with both packages,
    #   for the same disorder configuration
    chern_pythtb, ham_gap_pythtb = system_pytb.single_point_chern(
        formula="both", return_ham_gap=True
    )
    chern_tbmodels, ham_gap_tbmodels = system_tbm.single_point_chern(
        formula="both", return_ham_gap=True
    )

    if DEBUG_MODE and rank == 0:
        print(
            f"Rank {rank}: PythTB package, supercell size L = {L}, disorder strength "
            + f"= {w}, SPCN : {chern_pythtb['symmetric']}"
        )
        print(
            f"Rank {rank}: TBmodels package, supercell size L = {L}, disorder strength "
            + f"= {w}, SPCN : {chern_tbmodels['symmetric']}"
        )

    if rank == 0:
        assert math.isclose(
            chern_pythtb["asymmetric"], chern_tbmodels["asymmetric"], abs_tol=1e-10
        )
        assert math.isclose(
            chern_tbmodels["asymmetric"], 0.8854690224736179, abs_tol=1e-10
        )

        assert math.isclose(
            chern_pythtb["symmetric"], chern_tbmodels["symmetric"], abs_tol=1e-10
        )
        assert math.isclose(
            chern_tbmodels["symmetric"], 1.0026971589597036, abs_tol=1e-10
        )

        assert math.isclose(ham_gap_pythtb, ham_gap_tbmodels, abs_tol=1e-10)
        assert math.isclose(ham_gap_tbmodels, 6.296837373592453, abs_tol=1e-10)
