# StraWBerryPy
StraWBerryPy (Single-poinT and local invaRiAnts for Wannier Berriologies in Python) is a Python package calculating topological invariants for non-crystalline 2D topological insulators. The single-point formulas in the supercell framework for the Chern number [[Ceresoli-Resta](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.76.012405)] and for the spin Chern number [[Favata-Marrazzo](https://iopscience.iop.org/article/10.1088/2516-1075/acba6f/meta)] are implemented in StraWBerryPy. 
In addition, StraWBerryPy can handle finite systems (such as bounded samples and heterostructures) and compute the local Chern marker [[Bianco-Resta](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.84.241106)].
The code provides dedicated interfaces to tight-binding packages [PythTB](http://www.physics.rutgers.edu/pythtb/) and [Tbmodels](https://tbmodels.greschd.ch/en/latest/).


## Quick start
Here, two examples for calculating the single-point topological invariant in the supercell framework for tight-binding models in presence of Anderson disorder. 

### Kane-Mele example
Let's consider as prototype of quantum spin Hall insulator the Kane-Mele model and calculate the topological invariant through the single-point spin Chern number formulation.  

First create the tight-binding model in primitive cell using PythTB and Tbmodels packages. Create a supercell, for example including L lattice points in each lattice vector direction.  
Then add disorder in the supercell, for instance Anderson disorder which is a uniformly distributed random on site potential.  
For a reference implementation, look at [Kane-Mele example](strawberrypy/example_models/kane_mele.py).

```python
import numpy as np
from strawberrypy import single_point_spin_chern
from strawberrypy.example_models import kane_mele_pythtb, kane_mele_tbmodels, km_anderson_disorder_pythtb, km_anderson_disorder_tbmodels

L = 12              # supercell LxL linear size

# Topological phase
r = 1.              # r = rashba/spin_orb
e = 3.              # e = e_onsite/spin_orb
spin_o = 0.3        # spin_orb = spin_orb/t (t=1)

# # Trivial phase
# r = 3. 
# e = 5.5
# spin_o = 0.3

# Create Kane-Mele model in supercell LxL through PythTB package
km_pythtb_model = kane_mele_pythtb(r, e, spin_o, L)

# Create Kane-Mele model in supercell LxL through TBmodels package
km_tbmodels_model = kane_mele_tbmodels(r, e, spin_o, L)

w = 1.              # w = disorder strength such that random potential is in [-w/2, w/2]      

# Add Anderson disorder in PythTB model
np.random.seed(10)
km_pythtb_model = km_anderson_disorder_pythtb(km_pythtb_model, w)

# Add Anderson disorder in TBmodels model
np.random.seed(10)
km_tbmodels_model = km_anderson_disorder_tbmodels(km_tbmodels_model, w)
```
Finally, use single_point_spin_chern function to calculate the single-point invariant for the disordered model in the supercell framework.   
The function takes as inputs:
- the model created with PythTB or TBmodels packages 
- variable "spin" indicating which spin Chern number we want to calculate : 'up' or 'down'
- variable "formula" selecting which single-point formula we want to calculate : 'asymmetric', 'symmetric' or 'both'

```python
spin_chern = 'down'
which_formula = 'symmetric'

# Single Point Spin Chern Number (SPSCN) calculation for Pythtb model
spin_chern_pythtb = single_point_spin_chern(km_pythtb_model, spin=spin_chern, formula=which_formula)

# Single Point Spin Chern Number (SPSCN) calculation for TBmodels model
spin_chern_tbmodels = single_point_spin_chern(km_tbmodels_model, spin=spin_chern, formula=which_formula)

# If which_formula = 'both', then Single Point Spin Chern numbers are printed as follows : 'asymmetric' 'symmetric'
print('PythTB package, supercell size L =', L, ' disorder strength = ', w,  ' SPSCN :', *spin_chern_pythtb )
print('TBmodels package, supercell size L =', L, ' disorder strength = ', w,  ' SPSCN :', *spin_chern_tbmodels )
```

### Haldane example
The single-point invariant calculation can be also performed for a Chern insulator, like the Haldane model, using single_point_chern function.   
For the model implementation, look at [Haldane example](strawberrypy/example_models/haldane.py).
```python
import numpy as np
from strawberrypy import single_point_chern
from strawberrypy.example_models import haldane_pythtb, haldane_tbmodels, h_anderson_disorder_pythtb, h_anderson_disorder_tbmodels

L = 12              # supercell LxL linear size

# Topological phase
t = -4.                   # t = first neighbours real hopping
t2 = 1.                   # t2 = second neighbours hopping
delta = 2.                # delta = energy on site
phi = -np.pi/2.0          # phi = second neighbours hopping phase

# # Trivial phase
# t = -4.                   # t = first neighbours real hopping
# t2 = 1.                   # t2 = second neighbours hopping
# delta = 4.5               # delta = energy on site
# phi = -np.pi/6.0          # phi = second neighbours hopping phase
   
# Create Haldane model in supercell LxL through PythTB package
h_pythtb_model = haldane_pythtb(delta, t, t2, phi, L)

# Create Haldane model in supercell LxL through TBmodels package
h_tbmodels_model = haldane_tbmodels(delta, t, t2, phi, L)

w = 1.                      # w = disorder strength such that random potential is in [-w/2, w/2]

# Add Anderson disorder in PythTB model
np.random.seed(10)
h_pythtb_model = h_anderson_disorder_pythtb(h_pythtb_model, w)

# Add Anderson disorder in TBmodels model
np.random.seed(10)
h_tbmodels_model = h_anderson_disorder_tbmodels(h_tbmodels_model, w)

which_formula = 'both'

# Single Point Chern Number (SPCN) calculation for Pythtb model
chern_pythtb = single_point_chern(h_pythtb_model, formula=which_formula)

# Single Point Chern Number (SPCN) calculation for TBmodels model
chern_tbmodels = single_point_chern(h_tbmodels_model, formula=which_formula)

# If which_formula = 'both', then Single Point Chern Numbers are printed as follows : 'asymmetric' 'symmetric'
print('PythTB package, supercell size L =', L, ' disorder strength = ', w,  ' SPCN :', *chern_pythtb )
print('TBmodels package, supercell size L =', L, ' disorder strength = ', w,  ' SPCN :', *chern_tbmodels )
```
## Local invariants
Here an example for calculating the local Chern marker for a bounded Haldane model in presence of Anderson disorder. First create the tight-binding model in the primitive cell using PythTB and TBmodels packages. Then, the models has to be cut in both x and y directions specifying the size of the resulting samples.
To conclude the construction of the models the disorder can be added, for instance Anderson disorder which is a uniformly distributed random on site potential.
```python
import numpy as np
from strawberrypy import local_chern_marker, make_finite, onsite_disorder
from strawberrypy.example_models import haldane_pythtb, haldane_tbmodels

# Create Haldane models through PythTB and TBmodels packages
hmodel_pbc_tbmodels = haldane_tbmodels(delta = 0.5, t = -1, t2 = 0.15, phi = np.pi / 2, L = 1)
hmodel_pbc_pythtb = haldane_pythtb(delta = 0.5, t = -1, t2 = 0.15, phi = np.pi / 2, L = 1)

# Cut the models to make a sample of size 10 x 10
hmodel_obc_tbmodels = make_finite(model = hmodel_pbc_tbmodels, nx_sites = 10, ny_sites = 10)
hmodel_obc_pythtb = make_finite(model = hmodel_pbc_pythtb, nx_sites = 10, ny_sites = 10)

# Add Anderson disorder within [-w/2, w/2] to the samples. The argument spinstates specifies the spin of the model
hmodel_tbm_disorder = onsite_disorder(model = hmodel_tbm, w = 1, spinstates = 1, seed = 184)
hmodel_pythtb_disorder = onsite_disorder(model = hmodel_pythtb, w = 1, spinstates = 1, seed = 184)
```
Finally the local Chern marker can be computed using ```local_chern_marker```, which takes as arguments:
- the model created using PythTB or TBmodels (using ```model```)
- the size of the bounded sample (via ```nx_sites``` and ```ny_sites```)
- the direction along which to compute the local marker (if not specified otherwise the function computes the local marker for the whole lattice) using the argument ```direction```, and the cell along the orthogonal direction from which to start (via ```start```)
- a boolean value, ```return_projector```, which specifies if return also the ground state projector of the model, and the argument ```projector``` to input it in order to avoid recalculation of heavy objects
- the possibility to perform macroscopic averages for disordered samples, via the boolean ```macroscopic_average```, and the cutoff radius of the average using the argument ```cutoff```

In the example below, the local Chern marker is evaluated along the x direction, starting from the cell whose y reduced coordinate is half the dimension of the lattice, and is averaged over a region with cutoff 2:
```python
# Compute the local Chern markers for TBmodels and PythTB
lcm_tbmodels = local_chern_marker(model = hmodel_tbm_disorder, nx_sites = 10, ny_sites = 10, direction = 0, start = 5, macroscopic_average = True, cutoff = 2)
lcm_pythtb = local_chern_marker(model = hmodel_pythtb_disorder, nx_sites = 10, ny_sites = 10, direction = 0, start = 5, macroscopic_average = True, cutoff = 2)

print("Local Chern marker along the x direction starting from y=5, computed using TBmodels: ", lcm_tbmodels)
print("Local Chern marker along the x direction starting from y=5, computed using PythTB: ", lcm_pythtb)
```

## Installation
```
git clone https://github.com/strawberrypy-developers/strawberrypy.git
cd strawberrypy
pip install .
```

## Authors
Roberta Favata and Nicolas Baù and Antimo Marrazzo
