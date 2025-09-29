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

In particular, if you use the implementation of the single-point (Chern or ℤ₂) invariants
  
  R. Favata and A. Marrazzo
  Single-point spin Chern number in a supercell framework
  `Electronic Structure 5, 014005 (2023)`_

.. _Electronic Structure 5, 014005 (2023): https://doi.org/10.1088/2516-1075/acba6f

If you use the implementation of the local Chern marker in periodic boundary conditions:

  N. Baù and A. Marrazzo
  Local Chern marker for periodic systems
  `Phys. Rev. B 109, 014206 (2024)`_

.. _Phys. Rev. B 109, 014206 (2024): https://doi.org/10.1103/PhysRevB.109.014206

If you use the implementation of the local spin-Chern or the local ℤ₂ markers:

  N. Baù and A. Marrazzo
  Theory of local ℤ₂ topological markers for finite and periodic two-dimensional systems
  `Phys. Rev. B 110, 054203 (2024)`_

.. _Phys. Rev. B 110, 054203 (2024): https://doi.org/10.1103/PhysRevB.110.054203

If you use the implementation of the localization marker:

  A. Marrazzo and R. Resta
  A local theory of the insulating state
  `Phys. Rev. Lett. 122, 166602 (2019)`_
  
.. _Phys. Rev. Lett. 122, 166602 (2019): https://doi.org/10.1103/PhysRevLett.122.166602

Acknowledgements
++++++++++++++++
We acknowledge support from the `ICSC <https://www.supercomputing-icsc.it/en/icsc-home/>`_ – Centro Nazionale di Ricerca in High Performance Computing, Big Data and Quantum Computing, funded by European Union – `NextGenerationEU <https://next-generation-eu.europa.eu/index_en>`_ – `PNRR <https://www.italiadomani.gov.it/content/sogei-ng/it/it/home.html>`_, Missione 4 Componente 2 Investimento 1.4.

.. |pic1| image:: docs/_static/media/logoxweb.svg
  :width: 250
  :target: https://www.supercomputing-icsc.it/en/icsc-home/
.. |pic2| image:: docs/_static/media/Logo-Fin-Ngeu.png
  :width: 250
  :target: https://next-generation-eu.europa.eu/index_en
.. |pic3| image:: docs/_static/media/Logo_Italia_Domani.jpg
  :width: 250
  :target: https://www.italiadomani.gov.it/content/sogei-ng/it/it/home.html

|pic1| |pic2| |pic3|
