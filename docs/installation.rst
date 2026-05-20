.. role:: python(code)
    :language: python
    :class: highlight

Installation
============

To install StraWBerryPy, first clone the `GitHub repository <https://github.com/strawberrypy-developers/strawberrypy.git>`_ and navigate into the project directory:

.. code:: bash

   git clone https://github.com/strawberrypy-developers/strawberrypy.git
   cd strawberrypy

StraWBerryPy provides two installation methods: a **standard installation** and a **HPC installation** tailored for distributed memory systems and HPC clusters. To avoid conflicts, a virtual environment is recommended for both installation methods. You can create and activate a virtual environment using the following commands:

.. code:: bash

   python -m venv env
   source env/bin/activate

.. note:: Python 3.11 or higher is required for StraWBerryPy. Ensure you have the appropriate Python version installed and activated in your environment. If you want to work with a specific Python version, consider using tools like `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ to manage virtual environments with multiple Python versions.

Standard installation
---------------------
For standard use cases, you can build and install the serial version, relying on NumPy internal libraries and threads parallelization. This does not require any specialized external libraries. You can install it seamlessly using ``pip``:

.. code:: bash

   pip install .

Alternatively, you can use the provided Makefile:

.. code:: bash

   make install

HPC installation
----------------
To leverage high-performance computing (HPC) resources, StraWBerryPy supports distributed memory parallelization using MPI, ELPA, and ScaLAPACK via a custom Fortran kernel.

**Prerequisites.** Before proceeding with the HPC installation, ensure you have the following dependencies installed and properly configured on your system:

- *MPI*: install an MPI implementation such as OpenMPI or MPICH. Ensure that the MPI compiler wrappers (e.g., ``mpicc``, ``mpifort``) are available in your ``PATH``.
- *MKL library*: install the Intel MKL library for optimized linear algebra routines. Set the ``MKLROOT`` environment variable to point to the MKL installation directory.
- *ELPA library* (optional): install the `ELPA <https://elpa.mpcdf.mpg.de/software/index.html>`_ library for parallel eigenvalue computations. Set the environment variables ``ELPA_INC`` and ``ELPA_LIB`` to point to the ELPA include and library directories, respectively.

The MPI version automatically detects MPI availability and multi-core support at runtime and falls back to non-MPI (serial) execution if MPI is not detected. The MPI version requires BLAS, LAPACK, and ScaLAPACK libraries and mpi-wrapped compilers. Optionally, it can also leverage the ELPA library for parallel eigenvalue computations if ELPA is available and enabled.

**Using make (recommended).** You can compile and install the parallel version using ``make`` by explicitly passing the required library paths and flags to the Makefile. For example:

.. code:: bash

   make install USE_MPI=True USE_ELPA=False

.. warning:: You may need to adjust the library paths and flags in the Makefile depending on your system configuration.

**Using pip.** Alternatively, you can install the parallel version via ``pip`` by passing the required build options and environment variables directly in the command line. For example:

.. code:: bash

   CC=mpicc FC=mpifort FFLAGS=... LDFLAGS=... pip install . --config-settings=setup-args=...

Verifying the installation
--------------------------
To confirm that StraWBerryPy has been installed correctly, you can run the test suite provided in the repository. We recommend using ``pytest``:

.. code:: bash

   pip install pytest
   cd tests
   pytest -rP -v

To specifically verify that the HPC backend is functioning correctly, you can run a parallel test using ``mpirun`` or ``mpiexec``:

.. code:: bash

   pip install pytest-mpi
   cd tests
   mpirun -n 2 pytest --with-mpi -rP -v