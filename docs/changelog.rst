What's new
==========
In the following there is a short summary of the main changes in StraWBerryPy from the first release as `SPInv <https://github.com/roberta-favata/spinv>`_.

StraWBerry 0.3.0
^^^^^^^^^^^^^^^^

We modified the structure of the code to reduce the number of variables the user has to pass the functions. The basis class is ``Model``, from which ``Supercell`` and ``FiniteModel`` are derived. Using these classes is easier to both implement the funcions and use the code since the variables of the model are stored as attributes of the instances and are defined once at the beginning of the code upon creating the model. The previous functions are updated to work with this structure.

* Introduced three main classes ``Model``, ``Supercell`` and ``FiniteModel`` which are responsible for the creation of the specific model.
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
* Introduced the single-point Chern number and single-point spin-Chern number;