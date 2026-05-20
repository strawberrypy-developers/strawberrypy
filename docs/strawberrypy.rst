.. role:: python(code)
    :language: python
    :class: highlight

StraWBerryPy API
================

StraWBerryPy's API is organized in several modules, which are described in the following sections. The main module is :python:`strawberrypy`, which provides the main classes and functions to define models, compute topological invariants, and perform post-processing. The :mod:`strawberrypy.example_models <strawberrypy.example_models>` module contains predefined example models that can be used as templates to define custom models and to test the code. The :mod:`strawberrypy.postprocessing <strawberrypy.postprocessing>` module includes tools for post-processing the results of the calculations, such as plotting functions and correlation functions.

The central class of the code is the :python:`Model` class, which represents a tight-binding model in its unit cell. This can then be used to define finite or supercell models, which are represented by the :python:`FiniteModel` and :python:`Supercell` classes, respectively. These classes provide methods to compute both single-point topological invariants and local geometrical and topological quantities, as well as to add disorder and perform other operations on the model. The following sections provide a detailed description of the main modules and classes of StraWBerryPy.

.. automodule:: strawberrypy
   :no-members:

.. currentmodule:: strawberrypy

Model class
-----------

.. autosummary::
   :nosignatures:
   
   Model
   Model.add_disorder_bernoulli
   Model.add_disorder_uniform
   Model.add_vacancies
   Model.from_wannier90
   Model.make_finite
   Model.make_heterostructure
   Model.make_supercell

.. autoclass:: Model
   :members:
   :undoc-members:
   :show-inheritance:

Finite model subclass
---------------------

.. currentmodule:: strawberrypy.finite_model

.. autosummary::
   :nosignatures:
   
   FiniteModel
   FiniteModel.local_chern_marker
   FiniteModel.local_spin_chern_marker
   FiniteModel.local_z2_marker
   FiniteModel.localization_marker

.. autoclass:: FiniteModel
   :members:
   :undoc-members:
   :show-inheritance:

Supercell model subclass
------------------------

.. currentmodule:: strawberrypy.supercell

.. autosummary::
   :nosignatures:
   
   Supercell
   Supercell.pbc_local_chern_marker
   Supercell.pbc_local_spin_chern_marker
   Supercell.pbc_local_z2_marker
   Supercell.single_point_chern
   Supercell.single_point_spin_chern

.. autoclass:: Supercell
   :members:
   :undoc-members:
   :show-inheritance: