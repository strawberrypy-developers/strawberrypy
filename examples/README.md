# StrawberryPy examples

This directory contains a collection of example scripts and Jupyter Notebooks demonstrating how to use `strawberrypy` to model topological systems, apply disorder, and compute various topological invariants.

## List of Examples

### Python Scripts (MPI-enabled)
These scripts are designed to be run from the terminal, ideally using MPI to demonstrate parallel execution over multiple ranks. 
*Example usage:* `mpirun -n 4 python example_01.py`

* **`example_01.py` - Haldane model & Chern markers**  
  Demonstrates how to run the Haldane model with MPI parallelization. It creates a supercell, adds uniform Anderson disorder, and computes both the single-point Chern number and the PBC local Chern marker for the system.
* **`example_02.py` - Kane-Mele model & $\mathbb{Z}_2$ invariants**  
  Similar to the first example but focuses on spinful electrons. It sets up the Kane-Mele model, applies disorder, and calculates single-point spin-Chern numbers alongside the local spin-Chern marker and the local $\mathbb{Z}_2$ marker.

### Jupyter Notebooks
These notebooks offer an interactive, step-by-step walkthrough of `strawberrypy`'s features.

* **`example_03.ipynb` - Plotting local Chern markers**  
  Reproduces the Haldane model plot featured on the documentation's homepage. It introduces `strawberrypy.postprocessing` to visually map the PBC local Chern marker over the lattice. It also demonstrates how to extract the global topological invariant by tracing the bare local Chern marker and compares it with the single-point invariant.
* **`example_04.ipynb` - Phase diagrams**  
  Calculates the phase diagram to map the topological invariant against the disorder strength ($W/t$) for the Haldane model.
* **`example_05.ipynb` - Correlation functions**  
  Computes correlation functions for local markers and shows how to evalute critical exponents at the topological phase transition for the Haldane model.