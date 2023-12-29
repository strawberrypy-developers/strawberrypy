r"""
This module contains two equivalent functions that create an instance of the `Haldane model <https://doi.org/10.1103/PhysRevLett.61.2015>`_ with PythTB or TBmodels. The Haldane model is a tight-binding model describing spinless electrons hopping on a 2D honeycomb lattice with a staggered magnetic flux. The parameters of the model are the nearest-neighbor hopping :math:`t`, the on-site energy term :math:`\pm\Delta` with opposite signs on the two sublattices and the second-nearest-neighbor hopping term :math:`t_2e^{i\phi}`. The Hamiltonian of the model reads:

.. math::

    \mathcal{H} = \Delta\sum_{i}\left( c_{i,A}^{\dagger}c_{i,A}-c_{i,B}^{\dagger}c_{i,B} \right) + t \sum_{\langle ij\rangle}c_i^{\dagger}c_j+t_2\sum_{\langle\langle ij\rangle\rangle} e^{i\nu_{ij}\phi}c_i^{\dagger}c_j + \mathrm{h.c.}

where :math:`\nu_{ij}=\pm 1` is a factor accounting for the direction of the complex hopping.
"""

from pythtb import tb_model
from tbmodels import Model

import numpy as np

def haldane_pythtb(delta, t, t2, phi):
    # From http://www.physics.rutgers.edu/pythtb/examples.html#haldane-model
    lat=[[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
    orb=[[0.0,0.0],[1./3.,1./3.]]
    model=tb_model(2,2,lat,orb)
    model.set_onsite([-delta,delta])
    for lvec in ([ 0, 0], [-1, 0], [ 0,-1]):
        model.set_hop(t, 0, 1, lvec)
    for lvec in ([ 1, 0], [-1, 1], [ 0,-1]):
        model.set_hop(t2*np.exp(1.j*phi), 0, 0, lvec)
    for lvec in ([-1, 0], [ 1,-1], [ 0, 1]):
        model.set_hop(t2*np.exp(1.j*phi), 1, 1, lvec)

    return model

def haldane_tbmodels(delta, t, t2, phi):
    primitive_cell = [[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
    orb = [[0.0,0.0],[1./3.,1./3.]]
    h_model = Model(on_site=[-delta,delta], dim=2, occ=1, pos=orb, uc=primitive_cell)
    for lvec in ([0,0],[-1,0],[0,-1]):
        h_model.add_hop(t, 0,1,lvec)
    for lvec in ([1,0],[-1,1],[0,-1]):
        h_model.add_hop(t2*np.exp(1.j*phi), 0,0,lvec)
    for lvec in ([-1,0],[1,-1],[0,1]):
        h_model.add_hop(t2*np.exp(1.j*phi), 1,1,lvec) 

    return h_model