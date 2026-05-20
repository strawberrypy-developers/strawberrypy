.. role:: python(code)
    :language: python
    :class: highlight

What's new
==========
In the following, a short summary of the main changes in StraWBerryPy from its first release as `SPInv <https://github.com/roberta-favata/spinv>`_.

StraWBerryPy 0.4.0
^^^^^^^^^^^^^^^^^^

* The code is now parallelized with MPI and ScaLAPACK, allowing the calculation of quantities on large systems across multiple nodes.
* The class :python:`Model` has been updated to store the information of the input model in a more efficient way, and is now the object that should be called when creating the model, while :python:`Supercell` and :python:`FiniteModel` are spawned from it by calling the appropriate methods.
* The interface with `Wannier90 <https://wannier.org/>`_ has been updated to read the files also with TBmodels and PythTB.
* Introduced post-processing tools to compute averages in real space, trace of local topological markers in periodic boundary conditions, correlation functions and plot utilities.

StraWBerryPy 0.3.1
^^^^^^^^^^^^^^^^^^

* Introduced the local spin-Chern marker for finite and periodic system, as defined in Ref. `Baù-Marrazzo (2024b) <https://arxiv.org/abs/2404.04598>`_.
* Introduced the local :math:`\mathbb{Z}_{2}` marker for finite and periodic system, as defined in Ref. `Baù-Marrazzo (2024b) <https://arxiv.org/abs/2404.04598>`_.

StraWBerryPy 0.3.0
^^^^^^^^^^^^^^^^^^

We modified the structure of the code in order to make it much more user-friendly, reducing the number of variables the user has to pass to the functions and storing all the information from the input model at once at the beginning of the code upon creating the model. 
The base class is :python:`Model`, from which :python:`Supercell` and :python:`FiniteModel` are derived, and extrapolates all the needed information for performing the several calculations on the input model. All functionalities are provided as methods of  :python:`Supercell` and :python:`FiniteModel` classes.
The functions of previous versions have been updated in order to work with this new structure.

* Introduced three main classes :python:`Model`, :python:`Supercell` and :python:`FiniteModel` which are aimed at the creation of the specific model.
* Introduced the interface with `Wannier90 <https://wannier.org/>`_ through `WannierBerri <https://wannier-berri.org/index.html>`_, which allows to read Wannier Hamiltonians.
* Introduced the PBC local Chern marker for periodic system in a supercell, as defined in Ref. `Baù-Marrazzo (2024a) <https://doi.org/10.1103/PhysRevB.109.014206>`_.
* Introduced the localization marker for finite systems, as defined in Ref. `Marrazzo-Resta (2019) <https://doi.org/10.1103/PhysRevLett.122.166602>`_.
* Introduced the possibility of add disorder in the system both by an on-site uniformly distributed random term (Anderson disorder) and by vacancies in the lattice.

StraWBerryPy 0.2.0 (SPInv)
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Introduced :python:`make_finite` to remove periodic hoppings in the Hamiltonian.
* Introduced :python:`make_heterostructure` to create heterostructures.
* Introduced the Bianco-Resta local Chern marker, as defined in Ref. `Bianco-Resta (2011) <https://doi.org/10.1103/PhysRevB.84.241106>`_.

StraWBerryPy 0.1.0 (SPInv)
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Introduced `Haldane <https://doi.org/10.1103/PhysRevLett.61.2015>`_ and `Kane-Mele <https://doi.org/10.1103/PhysRevLett.95.226801>`_ models in :doc:`example_models<strawberrypy.example_models>`.
* Introduced the single-point Chern number and single-point spin Chern number, as defined in Refs. `Ceresoli-Resta (2007) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.76.012405>`_ and `Favata-Marrazzo (2023) <https://iopscience.iop.org/article/10.1088/2516-1075/acba6f/meta>`_.
