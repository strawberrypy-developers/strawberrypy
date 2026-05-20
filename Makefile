# --- Configuration & extras ---
USE_MPI  ?= false
USE_ELPA ?= false

# --- Toolchain definitions ---
CC  = mpicc
FC  = mpifort

# --- Dependency paths ---
ELPA_INC ?= ${ELPA_INC}
ELPA_LIB ?= ${ELPA_LIB}
MKLROOT  ?= ${MKLROOT}

# --- Compiler & linker flags ---
FFLAGS  += -O2 -march=native -I$(MKLROOT)/include
CFLAGS   = -O2 -march=native
CXXFLAGS = -O2 -march=native
LDFLAGS += -L$(MKLROOT)/lib \
           -lmkl_scalapack_lp64 -lmkl_gf_lp64 -lmkl_sequential \
           -lmkl_core -lmkl_blacs_openmpi_lp64 \
           -lpthread -lm -ldl

ifeq ($(USE_ELPA),true)
    FFLAGS  +=  -I$(ELPA_INC)
    LDFLAGS +=  -L$(ELPA_LIB) -lelpa 
endif

###############   DO NOT EDIT BELOW THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING   ###############

# Preserve any command-line `EXTRAS` and compute `PIP_EXTRAS`
EXTRAS ?=
PIP_EXTRAS := $(EXTRAS)
ifeq ($(USE_MPI),true)
    # If PIP_EXTRAS is empty, set it to 'mpi' (avoids leading comma)
    ifeq ($(strip $(PIP_EXTRAS)),)
        PIP_EXTRAS := mpi
    else
        # If PIP_EXTRAS has content, append ',mpi'
        PIP_EXTRAS := $(PIP_EXTRAS),mpi
    endif
endif

#    This prevents passing empty brackets .[] to pip if PIP_EXTRAS is empty
PIP_TARGET := .
ifneq ($(strip $(PIP_EXTRAS)),)
    PIP_TARGET := .[$(PIP_EXTRAS)]
endif

# Force source compilation of the mpi4py package to use correct external MPI library
PIP_TARGET_MPI := env MPICC=$(which mpicc) pip install --force-reinstall --no-binary=mpi4py mpi4py --no-cache-dir

# --- Targets ---

.PHONY: all install help

all: install

help:
	@echo "Available targets:"
	@echo "  make install  - Install package. (Current target: $(PIP_TARGET))"
	@echo "  make clean    - Remove build artifacts"

install:
	@echo "Installing $(PIP_TARGET) with MPI=$(USE_MPI) ELPA=$(USE_ELPA)"
	CC=$(CC) FC=$(FC) CFLAGS="$(CFLAGS)" CXXFLAGS="$(CXXFLAGS)" FFLAGS="$(FFLAGS)" LDFLAGS="$(LDFLAGS)" \
	pip install -v $(PIP_TARGET) \
	--config-settings=setup-args="-Duse_mpi=$(USE_MPI)" \
	--config-settings=setup-args="-Duse_elpa=$(USE_ELPA)"
    $(when $(USE_MPI), $(PIP_TARGET_MPI))

clean:
	rm -rf build/ dist/ *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +