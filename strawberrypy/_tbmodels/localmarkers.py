import tbmodels as tbm
import numpy as np
import scipy.linalg as la
from opt_einsum import contract

from .finite_systems import *
from strawberrypy.common_func import *

def local_chern_marker(model, nx_sites : int, ny_sites : int, direction : int = None, start : int = 0, return_projector : bool = False, projector : np.ndarray = None, macroscopic_average : bool = False, cutoff : float = 0.8):
    """
    Evaluate the Z Chern marker on the whole lattice if direction is None. If direction is not None evaluates the Z Chern marker along direction starting from start.
    
        Args:
        - model : instance of TBModels of the model
        - nx_sites : number of unit cells in the x direction of the finite system
        - ny_sites : number of unit cells in the y direction of the finite system
        - direction : direction along which compute the local Chern marker, default is None (returns the whole lattice Chern marker), allowed values are 0 for 'x' direction and 1 for 'y' direction
        - start : if direction is not None, is the coordinate of the unit cell at the start of the evaluation of the Chern marker
        - return_projector : if True, returns the ground state projector at the end of the calculation, default is False
        - projector : input the ground state projector to be used in the calculation. Default is None, which means it is computed from the model
        - macroscopic_average : if True, returns the local Chern marker averaged over a radius equal to the cutoff
        - cutoff : cutoff set for the calculation of averages

        Returns:
        - lattice_chern : local Chern marker of the whole lattice if direction is None
        - lcm_direction : local Chern marker along direction starting from start
        - projector : ground state projector, returned if return_projector is set True (default is False)
    """

    # Check input variables
    if direction not in [None, 0, 1]:
        raise RuntimeError("Direction allowed are None, 0 (which stands for x), and 1 (which stands for y)")
    
    if direction is not None:
        if direction == 0:
            if start not in range(ny_sites): raise RuntimeError("Invalid start parameter (must be within [0, ny_sites - 1])")
        else:
            if start not in range(nx_sites): raise RuntimeError("Invalid start parameter (must be within [0, nx_sites - 1])")

    # Atoms in the unit cell
    atoms_uc = int(model.size / (nx_sites * ny_sites))

    # 2D unit cell volume
    uc_vol = np.linalg.norm(np.cross(model.uc[0], model.uc[1]))

    if projector is None:
        # Eigenvectors at \Gamma
        hamiltonian = model.hamilton([0, 0], convention = 1)
        _, eigenvecs = la.eigh(hamiltonian)
        eigenvecs = eigenvecs

        # Build the ground state projector
        gs_projector = contract("ji,ki->jk", eigenvecs[:, :int(0.5 * len(eigenvecs))], eigenvecs[:, :int(0.5 * len(eigenvecs))].conjugate())
    else:
        gs_projector = projector

    # Convert TBM-positions in actual positions
    positions = get_positions(model, nx_sites, ny_sites)

    # Position operator in tight-binding approximation (site orbitals basis)
    rx = np.diag(positions[:, 0]); ry = np.diag(positions[:, 1])

    # Chern marker operator
    chern_operator = np.imag(gs_projector @ commutator(rx, gs_projector) @ commutator(ry, gs_projector))
    chern_operator *= -4 * np.pi / uc_vol

    # If macroscopic average I have to compute the lattice values with the averages first
    if macroscopic_average:
        chern_on_lattice = average_over_radius(np.diag(chern_operator), positions.T[0], positions.T[1], cutoff, nx_sites, ny_sites, atoms_uc)
    
    if direction is not None:
        # Evaluate index of the selected direction
        indices = xy_to_index('x' if direction == 1 else 'y', start, nx_sites, ny_sites, atoms_uc)

        # If macroscopic average consider the averaged lattice, else the Chern operator
        if macroscopic_average:
            lcm_direction = [chern_on_lattice[int(indices[atoms_uc * i] / atoms_uc)] for i in range(int(len(indices) / atoms_uc))]
        else:
            lcm_direction = [np.sum([chern_operator[indices[atoms_uc * i + j], indices[atoms_uc * i + j]] for j in range(atoms_uc)]) for i in range(int(len(indices) / atoms_uc))]
        
        if not return_projector:
            return np.array(lcm_direction)
        else:
            return np.array(lcm_direction), gs_projector

    if not macroscopic_average:
        chern_on_lattice = partialtrace(chern_operator, atoms_uc)

    lattice_chern = chern_on_lattice.reshape(nx_sites, ny_sites).T
    if not return_projector:
        return np.array(lattice_chern)
    else:
        return np.array(lattice_chern), gs_projector