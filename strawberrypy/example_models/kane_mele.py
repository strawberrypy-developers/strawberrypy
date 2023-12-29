r"""
This module contains two equivalent functions that create an instance of the `Kane-Mele model <https://doi.org/10.1103/PhysRevLett.95.226801>`_ with PythTB or TBmodels. The Kane-Mele model is a tight-binding model describing spinful electrons hopping on a 2D honeycomb lattice with spin-orbit coupling and a Rashba term (breaking :math:`S_z`-symmetry). The parameters of the model are the intensity of the diagonal spin-orbit coupling :math:`\lambda_{SO}`, the Rashba term :math:`\lambda_{R}`. The Hamiltonian of the model reads:

.. math::

    \mathcal{H} = \Delta\sum_{i}(-1)^{\tau_i}c_{i}^{\dagger}c_{i} + t \sum_{\langle ij\rangle}c_i^{\dagger}c_j+i\lambda_{SO}\sum_{\langle\langle ij\rangle\rangle} \nu_{ij}c_i^{\dagger}\sigma_zc_j + i\lambda_R\sum_{\langle ij\rangle}c_i^{\dagger}(\hat{\mathbf e}_{\langle ij\rangle}\cdot\boldsymbol\sigma)c_j + \mathrm{h.c.}
    
where :math:`\tau_i\in\{0,1\}` is an index which distinguishes the two sublattices, :math:`\nu_{ij}=\pm 1` accounts for the direction of the hoppings and :math:`\hat{\mathbf e}_{\langle ij\rangle}=\hat{\mathbf d}_{\langle ij\rangle}\times\hat{\mathbf z}` where :math:`\hat{\mathbf d}_{\langle ij\rangle}` is the unit vector in the direction from site :math:`i` to site :math:`j`. In the Hamiltonian, the double sum on the spin indices is implied in each term, with the convention that if no spin matrices appear, they are contracted over the identity.
"""

from pythtb import tb_model
from tbmodels import Model

import numpy as np

def kane_mele_pythtb(rashba, esite, spin_orb):
    # From http://www.physics.rutgers.edu/pythtb/examples.html#kane-mele-model-using-spinor-features

    # define lattice vectors
    lat=[[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
    # define coordinates of orbitals
    orb=[[0.0,0.0],[1./3.,1./3.]]

    # make two dimensional tight-binding Kane-Mele model
    km_model=tb_model(2,2,lat,orb,nspin=2)

    # set other parameters of the model
    thop=1.0

    rashba = rashba*spin_orb
    esite = esite*spin_orb

    # set on-site energies
    km_model.set_onsite([esite, -esite])

    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])

    # useful definitions
    sigma_x=np.array([0.,1.,0.,0])
    sigma_y=np.array([0.,0.,1.,0])
    sigma_z=np.array([0.,0.,0.,1])

    # spin-independent first-neighbor hops
    for lvec in ([ 0, 0], [-1, 0], [ 0,-1]):
        km_model.set_hop(thop, 0, 1, lvec)
        
    # spin-dependent second-neighbor hops
    for lvec in ([ 1, 0], [-1, 1], [ 0,-1]):
       km_model.set_hop(1.j*spin_orb*sigma_z, 0, 0, lvec)
    for lvec in ([-1, 0], [ 1,-1], [ 0, 1]):
       km_model.set_hop(1.j*spin_orb*sigma_z, 1, 1, lvec)

    # Rashba first-neighbor hoppings: (s_x)(dy)-(s_y)(d_x)
    r3h =np.sqrt(3.0)/2.0

    # bond unit vectors are (r3h,half) then (0,-1) then (-r3h,half)
    km_model.set_hop(1.j*rashba*( 0.5*sigma_x-r3h*sigma_y), 0, 1, [ 0, 0], mode="add")
    km_model.set_hop(1.j*rashba*(-1.0*sigma_x            ), 0, 1, [ 0,-1], mode="add")
    km_model.set_hop(1.j*rashba*( 0.5*sigma_x+r3h*sigma_y), 0, 1, [-1, 0], mode="add")

    return km_model
   
def kane_mele_tbmodels(rashba,esite,spin_orb):

    t_hop = 1.0
    r = rashba*spin_orb
    e = esite*spin_orb

    primitive_cell = [[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
    orb = [[0.0,0.0],[0.0,0.0],[1./3.,1./3.],[1./3.,1./3.]]

    #set energy on site
    km_model = Model(on_site=[e,e,-e,-e], dim=2, occ=2, pos=orb, uc=primitive_cell)

    #spin-independent first-neighbor hops
    for lvec in ([0,0],[-1,0],[0,-1]):
        km_model.add_hop(t_hop,0,2,lvec)
        km_model.add_hop(t_hop,1,3,lvec)

    #spin-dependent second-neighbor hops
    for lvec in ([1,0],[-1,1],[0,-1]):
        km_model.add_hop(1.j*spin_orb, 0,0,lvec)
        km_model.add_hop(-1.j*spin_orb, 1,1,lvec)
    for lvec in ([-1,0],[1,-1],[0,1]):
        km_model.add_hop(1.j*spin_orb,2,2,lvec)
        km_model.add_hop(-1.j*spin_orb, 3,3,lvec)

    #Rashba first neighbor hoppings : (s_x)(dy)-(s_y)(dx)
    r3h = np.sqrt(3.0)/2.0

    #bond unit vectors are (r3h,half) then (0,-1) then (-r3h,half)
    km_model.add_hop(1.j*r*(0.5-1.j*r3h), 1, 2, [ 0, 0])
    km_model.add_hop(1.j*r*(0.5+1.j*r3h), 0, 3, [ 0, 0])
    km_model.add_hop(-1.j*r, 1, 2, [ 0, -1])
    km_model.add_hop(-1.j*r, 0, 3, [ 0, -1])
    km_model.add_hop(1.j*r*(0.5+1.j*r3h), 1, 2, [ -1, 0])
    km_model.add_hop(1.j*r*(0.5-1.j*r3h), 0, 3, [ -1, 0])

    return km_model