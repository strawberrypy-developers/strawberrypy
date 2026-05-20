import numpy as np

from ..config import mpimaster
from ..classes import Model
from .contractions import lattice_contraction, pbc_lattice_contraction
from .averages import average_over_radius


@mpimaster
def trace_marker(
    model: Model,
    marker: np.ndarray,
    invariant: str = "chern",
    bare_marker: bool = False,
    macroscopic_average: bool = False,
    cutoff: float = 0.8,
    boundary_conditions: str = "pbc",
) -> float:
    r"""
    Trace a topological marker over the whole supercell to get the single-point invariant.
    This function is intended for periodic boundary conditions (PBC) only since within open
    boundary conditions the trace over the whole sample vanishes by construction.

    Parameters
    ----------
        model : Model
            The model for which to compute the topological invariant.
        marker : np.ndarray
            The PBC local topological marker to be traced.
        invariant : str
            The type of topological marker computed. Allowed values are :python:`'chern'`
            when passing the PBC local Chern marker, and :python:`'z2'` when passing the
            PBC local spin-Chern marker or the local :math:`\mathbb{Z}_{2}` marker using
            time reversal symmetry.
        bare_marker : bool
            If the local marker passed is the bare marker. Default is :python:`False`.
        macroscopic_average : bool
            If :python:`True`, performs a real space average of the local marker over a
            radius equal to :python:`cutoff`, and then trace over the whole sample. :python:`marker` is expected to be the bare marker in this case and :python:`bare_marker` should be set to :python:`True`.
            Default is :python:`False`.
        cutoff : float
            Cutoff set for the calculation of the macroscopic average in real space.
        boundary_conditions : str
            If :python:`macroscopic_average == True`, the boundary conditions to be used
            when computing the macroscopic average in real space. Default is :python:`pbc`.

    Returns
    -------
        sp_inv : float
            Topological invariant computed as the trace over the whole supercell of the
            local topological marker.
    """
    # Only the master rank in parallel can call functions distributing the computation
    master_rank = model.backend.is_master_rank

    # Check input variables
    if invariant not in ("chern", "z2"):
        raise RuntimeError(
            "Invariant type not allowed. Allowed values are 'chern' and 'z2'."
        )
    if macroscopic_average and not bare_marker:
        raise RuntimeError(
            "Macroscopic average can be performed only if the bare marker is passed."
        )

    if bare_marker and macroscopic_average:
        # If macroscopic_average, prepare the contraction window for the average in real space
        if boundary_conditions == "pbc":
            contraction = pbc_lattice_contraction(model, cutoff)
        elif boundary_conditions == "obc":
            contraction = lattice_contraction(model, cutoff)
        else:
            raise RuntimeError(
                "Boundary condition for the macroscopic average not allowed."
            )
            
        if invariant == "chern":
            marker = average_over_radius(
                    model, marker, cutoff, contraction=contraction
                )
        else:
            marker[0] = average_over_radius(
                model, marker[0], cutoff, contraction=contraction
            )
            marker[1] = average_over_radius(
                model, marker[1], cutoff, contraction=contraction
            )
            
    if master_rank:
        # Trace over the whole sample and set the correct normalization factor
        if invariant == "chern":
            if bare_marker and not macroscopic_average:
                sp_inv = np.sum(marker)  * (model.states_uc / model.n_orb)
            else: 
                sp_inv = np.sum(marker) / (model.n_orb)
                
        else:
            if bare_marker and not macroscopic_average:
                sp_c1 = np.sum(marker[0]) * (model.states_uc / model.n_orb)
                sp_c2 = np.sum(marker[1]) * (model.states_uc / model.n_orb)
            else:
                sp_c1 = np.sum(marker[0]) / (model.n_orb)
                sp_c2 = np.sum(marker[1]) / (model.n_orb)

            # sp_inv = np.abs(np.fmod(0.5 * (sp_c1 - sp_c2), 2))
            sp_inv = np.mod(0.5 * (sp_c1 - sp_c2), 2)
            sp_inv = 1 - np.abs(1 - sp_inv)

    sp_inv = model.backend.comm.bcast(sp_inv if master_rank else None, root=0)
    return sp_inv
