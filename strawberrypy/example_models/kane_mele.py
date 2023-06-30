import numpy as np
from tbmodels import *
from pythtb import *

def kane_mele_pythtb(rashba, esite, spin_orb, L):
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

    sc_model = km_model.make_supercell([[L,0],[0,L]])

    return sc_model
   
def kane_mele_tbmodels(rashba,esite,spin_orb,L):

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

    sc_model = km_model.supercell([L,L])
    return sc_model