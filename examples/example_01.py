r"""Running the Haldane model with MPI

In this example, we are running the Haldane model with MPI parallelization to compute
  single-point Chern number, and the PBC local Chern marker.

You can run this example with: `mpirun -n 4 python example_01.py`
"""

import numpy as np
import strawberrypy as strw

# Parameters of the model and the supercell
L = 12
delta = 1
t2 = 1
t = 3
phi = np.pi / 2
seed = 12345

# Define the model and make a supercell
h_model = strw.example_models.haldane_tbmodels(delta, t, t2, phi)
hmodel_sc = strw.Model(model=h_model, spinful=False).make_supercell(Lx=L, Ly=L)

# Add Anderson disorder
hmodel_sc.add_disorder_uniform(w=10, seed=seed)

# Enable debug info for printing
hmodel_sc.debug_info = True

# Compute the single-point Chern number and the local Chern marker
sp = hmodel_sc.single_point_chern(formula="both", return_ham_gap=True)
lcm = hmodel_sc.pbc_local_chern_marker(
    bare_marker=False,
    smearing_temperature=0.1,
    n_tba=1,
    macroscopic_average=True,
    cutoff=0.7,
)

# Note that relevant quantities are defined only on the master rank
if hmodel_sc.backend.is_master_rank:
    print("\nResults:")
    print()
    print(f"  Symmetric single-point Chern number: {sp[0]['symmetric']}")
    print(f"  Asymmetric single-point Chern number: {sp[0]['asymmetric']}")
    print(f"  Gap of the Hamiltonian: {sp[1]}")
    print()
    print(f"  Local Chern marker:")
    print(f"    {lcm[:10]}")
    print()
    print("  Trace of the local markers:")
    print(f" {strw.postprocessing.trace_marker(hmodel_sc, lcm, bare_marker=False)}")
