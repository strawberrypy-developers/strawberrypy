r"""Running the Kane-Mele model with MPI

In this example, we are running the Kane-Mele model with MPI parallelization to compute
  single-point spin-Chern number, as well as the local spin-Chern and Z2 markers.
  The example is similar to the Haldane model example, but with the Kane-Mele
  model and spinful electrons.

You can run this example with: `mpirun -n 4 python example_02.py`
"""

import strawberrypy as strw

# Parameters of the model and the supercell
L = 8
r = 1.0
e = 3.0
spin_o = 0.3
seed = 12345

# Define the model and make a supercell
km = strw.example_models.kane_mele_tbmodels(r, e, spin_o)
model_sc = strw.Model(model=km, spinful=True).make_supercell(Lx=L, Ly=L)

# Add Anderson disorder
model_sc.add_disorder_uniform(w=0.1, seed=seed)

# Enable debug info for printing
model_sc.debug_info = True

# Compute the single-point spin-Chern numbers for up and down spins
spup = model_sc.single_point_spin_chern(
    spin="up", formula="both", return_pszp_gap=True, return_ham_gap=True
)
spdw = model_sc.single_point_spin_chern(
    spin="down", formula="both", return_pszp_gap=True, return_ham_gap=True
)

# Compute the local spin-Chern marker and the local Z2 marker
lscm = model_sc.pbc_local_spin_chern_marker(smearing_temperature=0.0, bare_marker=True)
lz2 = model_sc.pbc_local_z2_marker(smearing_temperature=0.0, bare_marker=True)

# Note that relevant quantities are defined only on the master rank
if model_sc.backend.is_master_rank:
    print("\nResults:")
    print()
    print(f"  Single-point spin-Chern number (up): {spup}")
    print(f"  Single-point spin-Chern number (down): {spdw}")
    print()
    print("  Local spin-Chern marker:")
    print(f"    {lscm[:, :6]}")
    print("  Local Z2 marker:")
    print(f"    {lz2[:, :6]}")
    print()
    print("  Trace of the local markers:")
    print(f"    Spin (up): {strw.postprocessing.trace_marker(model_sc, lscm[0], bare_marker=True)}")
    print(f"    Spin (down): {strw.postprocessing.trace_marker(model_sc, lscm[1], bare_marker=True)}")
    print()
    print(f"    Z2 (subspace 1): {strw.postprocessing.trace_marker(model_sc, lz2[0], bare_marker=True)}")
    print(f"    Z2 (subspace 2): {strw.postprocessing.trace_marker(model_sc, lz2[1], bare_marker=True)}")
