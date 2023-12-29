What's new
==========
In the following, a short summary of the main changes in StraWBerryPy from its first release as `SPInv <https://github.com/roberta-favata/spinv>`_.

StraWBerry 0.3.0
^^^^^^^^^^^^^^^^

We modified the structure of the code in order to make it much more user-friendly, reducing the number of variables the user has to pass to the functions and storing all the information from the input model at once at the beginning of the code upon creating the model. 
The base class is ``Model``, from which ``Supercell`` and ``FiniteModel`` are derived, and extrapolates all the needed information for performing the several calculations on the input model. All functionalities are provided as methods of  ``Supercell`` and ``FiniteModel`` classes.
The functions of previous versions have been updated in order to work with this new structure.

* Introduced three main classes ``Model``, ``Supercell`` and ``FiniteModel`` which are aimed at the creation of the specific model.
* Introduced the localization marker for finite systems.
* Introduced the PBC local Chern marker for periodic system in a supercell.
* Introduced the possibility of add disorder in the system both by an on-site uniformly distributed random term (Anderson disorder) and by vacancies in the lattice.
* Introduced the interface with `Wannier90 <https://wannier.org/>`_ through `WannierBerri <https://wannier-berri.org/index.html>`_, which allows to read Wannier Hamiltonians.

StraWBerryPy 0.2.0 (SPInv)
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Introduced ``make_finite`` to remove periodic hoppings in the Hamiltonian;
* Introduced ``make_heterostructure`` to create heterostructures;
* Introduced the Bianco-Resta local Chern marker;

StraWBerryPy 0.1.0 (SPInv)
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Introduced Haldane and Kane-Mele example models;
* Introduced the single-point Chern number and single-point spin Chern number;