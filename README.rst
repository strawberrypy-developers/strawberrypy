============
StraWBerryPy
============

**StraWBerryPy** (Single-poinT and local invaRiAnts for Wannier Berriologies in Python) is a Python package to calculate topological invariants and quantum-geometrical quantities in non-crystalline topological insulators.

The code reads tight-binding models from `PythTB <http://www.physics.rutgers.edu/pythtb/>`_, `TBmodels <https://tbmodels.greschd.ch/en/latest/>`_ and `Wannier90 <https://wannier.org/>`_ (through `WannierBerri <https://wannier-berri.org/index.html>`_). 

StraWBerryPy can work both with periodic (PBCs) and open (OBCs) boundary conditions. The code allows to create and manipulate supercells and finite models, for example adding disorder. Single-point and local topological markers can be computed, in addition to other quantum-geometrical quantities (e.g., the localization marker).

* `Github page <https://github.com/strawberrypy-developers/strawberrypy>`_
* `Documentation <http://strawberrypy.readthedocs.io/>`_


How to cite
+++++++++++
Please cite the following papers in any publication arising from the use of 
this code. 

In particular, if you use the implementation of the single-point (Chern or Z2) invariants
  
  R. Favata and A. Marrazzo
  Single-point spin Chern number in a supercell framework
  `Electronic Structure 5, 014005 (2023)`_

.. _Electronic Structure 5, 014005 (2023): https://doi.org/10.1088/2516-1075/acba6f

If you use the implementation of the local Chern marker in periodic boundary conditions:

  N. Baù and A. Marrazzo
  Local Chern marker for periodic systems
  `accepted in PRB, arxiv (2023)`_

.. _accepted in PRB, arxiv (2023): https://doi.org/10.48550/arXiv.2310.15783

If you use the implementation of the localization marker:

  A. Marrazzo and R. Resta
  A local theory of the insulating state
  `Phys. Rev. Lett. 122, 166602 (2019)`_
  
.. _Phys. Rev. Lett. 122, 166602 (2019): https://doi.org/10.1103/PhysRevLett.122.166602

