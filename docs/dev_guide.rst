.. role:: python(code)
    :language: python
    :class: highlight
    
Developer manual
================

This page provides guidelines for developers contributing to StraWBerryPy. It covers the development setup, repository structure, testing, and code style.

Setting up for development
--------------------------
To develop StraWBerryPy, we recommend using a virtual environment, install the package in editable mode and include the development dependencies (like testing, linting and sphinx tools):

.. code:: bash

    git clone https://github.com/strawberrypy-developers/strawberrypy.git
    cd strawberrypy
    pip install meson-python
    pip install -e .[mpi,devs] --no-build-isolation [--other-args]

where the external dependencies (like BLAS, LAPACK, ScaLAPACK, ELPA) should be provided to compile the Fortran extensions. For instance, if using Intel MKL with OpenMPI and ELPA, the command could be:

.. code:: bash

    export ELPA_ROOT=$HOME/usr/elpa
    export ELPA_LIB=$ELPA_ROOT/lib
    export ELPA_INC=$ELPA_ROOT/include/elpa-2025.06.002/modules
    export LD_LIBRARY_PATH=$ELPA_LIB:$LD_LIBRARY_PATH

    CC=mpicc FC=mpifort FFLAGS="-I${ELPA_INC}               \
        -I${MKLROOT}/include" LDFLAGS="-L${ELPA_LIB}        \
        -lelpa -L${MKLROOT}/lib -lmkl_scalapack_lp64        \
        -lmkl_gf_lp64 -lmkl_sequential -lmkl_core           \
        -lmkl_blacs_openmpi_lp64 -lpthread -lm -ldl"        \
        pip install -v -e .[mpi,devs] --no-build-isolation  \
        --config-settings=setup-args="-Duse_mpi=true"       \
        --config-settings=setup-args="-Duse_elpa=true"

When installing, make sure all dependencies are compiled with the same compilers. Otherwise, there will be dependency errors. A debug mode, returning more verbose output, can be enabled by setting the environment variable ``STRAWBERRYPY_DEBUG_MODE`` to ``1``, even at runtime. This can be useful for debugging and development purposes.

Sample installation of external dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If needed, one can use SPACK to install all dependencies with the same compilers.

.. code:: bash
    
    git clone --depth=2 --branch=releases/v1.1 https://github.com/spack/spack.git ~/spack
    cd ~/spack

    source share/spack/setup-env.sh

    # Install compilers, MPI, and MKL with SPACK
    spack install gcc@11.4.0
    spack install openmpi@5.0.8%gcc@11.4.0
    spack install intel-oneapi-mkl%gcc@11.4.0

    # Load the installed dependencies into the environment
    spack load gcc@11.4.0 openmpi@5.0.8%gcc@11.4.0 intel-oneapi-mkl%gcc@11.4.0

    # Install ELPA
    wget https://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/2026.02.001/elpa-2026.02.001.tar.gz
    tar zxvf elpa-2026.02.001.tar.gz
    cd elpa-2026.02.001
    mkdir build && cd build
    export prefix=$HOME/usr/elpa
    ../configure --prefix=${prefix} CC=mpicc FC=mpifort CXX=mpicxx              \
        --enable-option-checking=fatal --disable-avx512-kernels                 \
        CFLAGS="-O2 -march=native" CXXFLAGS="-O2 -march=native"                 \
        FCFLAGS="-O2 -march=native" SCALAPACK_LDFLAGS="-m64 -L${MKLROOT}/lib    \
        -lmkl_scalapack_lp64 -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_sequential  \
        -lmkl_core -lmkl_blacs_openmpi_lp64 -lpthread -lm -ldl"                 \
        SCALAPACK_FCFLAGS="-m64 -I${MKLROOT}/include" 2>&1 | tee config.out
    make -j8 2>&1 | tee make.out
    make install 2>&1 | tee install.out

Please consult `MKL link line advisor <https://www.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html>`_ on how to setup proper flags for your specific system.

Repository and code structure
-----------------------------
The repository is organized as follows:

.. code:: bash

    strawberrypy/
    ├── docs/               # Documentation source files
    ├── examples/           # Example scripts and notebooks
    ├── strawberrypy/       # Main package source code
    │   ├── __init__.py
    │   ├── _[3rd party]    # 3rd party parsers (e.g., PythTB, TBmodels, WannierBerri)
    │   ├── backends/       # Serial and parallel backends
    │   ├── example_models/ # Example models for testing and demonstration
    │   ├── postprocessing/ # Post-processing tools
    │   ├── classes.py      # Core Model class
    │   └── ...             # Other submodules
    ├── tests/              # Tests
    ├── setup.py            # Installation script
    ├── Makefile            # Installation script
    ├── meson.build         # Build configuration for Fortran extensions
    ├── meson_options.txt   # Build options for Fortran extensions
    ├── README.md           # Project overview and instructions
    └── ...                 # Other files (e.g., LICENSE, .gitignore, ...)

The main class is ``Model``, which serves as the core data structure for representing tight-binding models in StraWBerryPy. It provides methods for parsing models from 3rd party formats, compute derived classes and modifications to th esystem (like vacancies and disorder). New features related to the model and shared by all the subclasses (``Supercell``, ``FiniteModel``, ...) should be implemented in the ``Model`` class, while features specific to a particular subclass should be implemented in the corresponding file. If new subclasses are added, they should inherit from the ``Model`` class, which should have methods to spawn the new subclass. If any new file is added, it should be added to the ``meson.build`` file to be included in the build process.

Serial and parallel backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The backends (serial and parallel) are implemented in separate files under the ``backends/`` directory, and they provide the computational routines for linear algebra and physics. In the same folder, the Fortran extensions are implemented to be compiled and imported by the parallel backend. Numpy F2PY (Fortran to Python interface generator) is used to create Python bindings for the Fortran code, and the build process is configured using Meson. This enables calling Fortran routines directly from Python, allowing for efficient computation while maintaining the flexibility of Python for higher-level operations. The directory is structured into two main components:

* *Serial backend* (``serial_linalg.py``, ``serial_physics.py``): implements the computational operations using conventional single-node Python libraries, such as ``numpy`` and ``scipy.linalg``. It implements dummy ``MPI`` and ``COMM_WORLD`` classes, allowing the rest of the codebase to call MPI-like methods (like ``bcast`` and ``gather``) transparently without requiring heavy conditional branches.
* *Parallel backend* (``mpi_linalg.py``, ``mpi_physics.py``): designed for HPC using MPI. It utilizes ``mpi4py`` to manage communication between processes and distributes matrices across a virtual grid of processors in a block-cyclic manner to minimize memory overhead. The heavy lifting for distributed linear algebra operations (such as diagonalizations via ELPA or ScaLAPACK) is dispatched into the compiled Fortran extensions.

Both backends are designed to share identical function signatures for their corresponding physics and linear algebra classes. This unified interface ensures that switching a model between serial and parallel mode is handled automatically to the user based on the execution environment and initialization backend.

In the backends, custom functions to create, distribute and gather matrices in block-cyclic layout are implemented. These can be accessed via the ``Model.backend`` attribute. In a serial execution, these functions are implemented as dummy functions that simply return the input, while in a parallel execution, they will perform the necessary communication and data distribution. The following functions are available for matrix distribution and gathering:

.. list-table::
   :header-rows: 0
   :class: autosummary longtable

   * - :obj:`distribute <strawberrypy.backends.mpi_linalg.MPIRoutine.distribute>`
     - Distribute a matrix defined on a specific rank to all processes using a ScaLAPACK-style block-cyclic layout. Return the local portion of the matrix on each process in parallel, or the full matrix on the root process in serial.
   * - :obj:`distribute_diag <strawberrypy.backends.mpi_linalg.MPIRoutine.distribute_diag>`
     - Distribute a 1D array as the diagonal of a global matrix in block-cyclic layout. Return the 2D local portion of the matrix on each process in parallel, or the full matrix on the root process in serial.
   * - :obj:`gather <strawberrypy.backends.mpi_linalg.MPIRoutine.gather>`
     - Gather local block-cyclic pieces from all ranks and reconstruct the global matrix on a given rank. Return the full matrix on the root process in serial.
   * - :obj:`get_diag <strawberrypy.backends.mpi_linalg.MPIRoutine.get_diag>`
     - Extract diagonal elements of a global matrix that is stored distributed in block-cyclic layout. Return the full diagonal as a 1D array on the root process in serial.
   * - :obj:`shared_array <strawberrypy.backends.mpi_linalg.MPIRoutine.shared_array>`
     - Create a node-local shared array from data owned by the root rank. In serial, return a NumPy array.

Via the same attribute ``Model.backend``, the following linear algebra functions are available for use in the codebase, with the same function signatures in both serial and parallel backends:

.. currentmodule:: strawberrypy.backends.mpi_linalg.MPILinalg

.. autosummary::
    :nosignatures:

    trace
    eigh
    matmul
    commutator
    linsolve

The physics backend, accessible via the attribute ``Model.physics``, implement the following functions:

.. currentmodule:: strawberrypy.backends.mpi_physics.MPIPhysics

.. autosummary::
    :nosignatures:

    fermidirac
    chemical_potential
    smearing
    get_proj

In all the previous modules, private functions (with a leading underscore) are not included in the documentation, but they are still available for use in the codebase. If you need to use a private function, please inspect its documentation in the `source files <https://github.com/strawberrypy-developers/strawberrypy/tree/main/strawberrypy/backends>`_ before using it.

Testing
-------
We use ``pytest`` for unit testing. Before submitting any changes, ensure all tests pass. To run the standard serial tests:

.. code:: bash

    cd tests/
    pytest -rP -v

To run the parallel tests (requires HPC installation):

.. code:: bash

    cd tests/
    mpirun -n 2 pytest --with-mpi -rP -v

**Writing new tests:** if you add a new feature, please add a corresponding test in the ``tests/`` directory.

Code style
----------
We adhere to standard coding guidelines to maintain code quality:

* *Python:* we use ``black`` formatter (with options ``-l 90 -C``) before committing.
* *Docstrings:* we use the [NumPy / Sphinx] docstring format. All functions and classes must be documented.

Building the documentation
--------------------------
The documentation is built using Sphinx. To build the documentation locally and preview your changes:

.. code:: bash

    cd docs/
    make clean html

You can then open ``docs/_build/html/index.html`` in your web browser.