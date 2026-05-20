import math
import pytest

from strawberrypy import Model, example_models
from strawberrypy.config import DEBUG_MODE


@pytest.mark.mpi
def test_spscn(L=6, r=1.0, e=3.0, spin_o=0.3, w=2.0):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # inputs are:    linear size of supercell LxL
    #               r = rashba/spin_orb
    #               e = e_onsite/spin_orb
    #               w = disorder stregth W/t
    #               spin_chern = choice of spin Chern number  'up' or 'down'
    #               which_formula = choice of single point formula 'asymmetric', 'symmetric' or 'both'

    # create Kane-Mele model in the primitive cell through PythTB package
    km_pythtb = example_models.kane_mele_pythtb(r, e, spin_o)

    # create Kane-Mele model in the primitive cell through TBmodels package
    km_tbmodels = example_models.kane_mele_tbmodels(r, e, spin_o)

    # initialize supercell models
    system_pytb = Model(km_pythtb, spinful=True).make_supercell(Lx=L, Ly=L)
    system_tbm = Model(km_tbmodels, spinful=True).make_supercell(Lx=L, Ly=L)

    # add Anderson disorder
    system_tbm.add_disorder_uniform(w, seed=10)
    system_pytb.add_disorder_uniform(w, seed=10)

    # Single Point Spin Chern Number (SPSCN) calculation for models created with
    #   both packages, for the same disorder configuration
    spin_chern_pythtb, pszp_gap_pythtb, ham_gap_pythtb = (
        system_pytb.single_point_spin_chern(
            formula="both", return_pszp_gap=True, return_ham_gap=True
        )
    )
    spin_chern_tbmodels, pszp_gap_tbmodels, ham_gap_tbmodels = (
        system_tbm.single_point_spin_chern(
            formula="both", return_pszp_gap=True, return_ham_gap=True
        )
    )

    if DEBUG_MODE and rank == 0:
        print(
            f"Rank {rank}: PythTB package, supercell size L = {L}, disorder strength "
            + f"= {w}, SPSCN : {spin_chern_pythtb['symmetric']}"
        )
        print(
            f"Rank {rank}: TBmodels package, supercell size L = {L}, disorder strength "
            + f"= {w}, SPSCN : {spin_chern_tbmodels['symmetric']}"
        )

    if rank == 0:
        assert math.isclose(
            spin_chern_pythtb["asymmetric"],
            spin_chern_tbmodels["asymmetric"],
            abs_tol=1e-10,
        )
        assert math.isclose(
            spin_chern_tbmodels["asymmetric"], 0.847124187954354, abs_tol=1e-10
        )

        assert math.isclose(
            spin_chern_pythtb["symmetric"],
            spin_chern_tbmodels["symmetric"],
            abs_tol=1e-10,
        )
        assert math.isclose(
            spin_chern_tbmodels["symmetric"], 1.0098001964392742, abs_tol=1e-10
        )

        assert math.isclose(ham_gap_pythtb, ham_gap_tbmodels, abs_tol=1e-10)
        assert math.isclose(ham_gap_tbmodels, 1.0201819613659904, abs_tol=1e-10)

        assert math.isclose(pszp_gap_pythtb, pszp_gap_tbmodels, abs_tol=1e-10)
        assert math.isclose(pszp_gap_tbmodels, 1.740912546954708, abs_tol=1e-10)
