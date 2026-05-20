.. role:: python(code)
    :language: python
    :class: highlight

Predefined example models
=========================

A collection of predefined example models is provided in the :python:`strawberrypy.example_models` module, that can be used as templates to define custom models and to test the code. The module includes the Haldane and Kane-Mele models, which are paradigmatic models of Chern and :math:`\mathbb{Z}_{2}` topological insulators, respectively.

.. automodule:: strawberrypy.example_models
   :no-members:

Haldane model
~~~~~~~~~~~~~

.. automodule:: strawberrypy.example_models.haldane
   :no-members:

.. currentmodule:: strawberrypy.example_models

.. autosummary::
   
    haldane_tbmodels
    haldane_pythtb

.. py:function:: haldane_tbmodels(delta, t, t2, phi)
.. py:function:: haldane_pythtb(delta, t, t2, phi, ptb_version='2.0.0')

Kane-Mele model
~~~~~~~~~~~~~~~

.. automodule:: strawberrypy.example_models.kane_mele
   :no-members:

.. currentmodule:: strawberrypy.example_models

.. autosummary::
   
    kane_mele_tbmodels
    kane_mele_pythtb

.. py:function:: kane_mele_tbmodels(rashba, esite, spin_orb)
.. py:function:: kane_mele_pythtb(rashba, esite, spin_orb, ptb_version='2.0.0')