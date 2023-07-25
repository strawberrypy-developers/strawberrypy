from pythtb import tb_model
from tbmodels import Model

import numpy as np

def haldane_pythtb(delta, t, t2, phi, L):
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

    sc_model = model.make_supercell([[L,0],[0,L]])
    return sc_model

def haldane_tbmodels(delta, t, t2, phi, L):
    primitive_cell = [[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
    orb = [[0.0,0.0],[1./3.,1./3.]]
    h_model = Model(on_site=[-delta,delta], dim=2, occ=1, pos=orb, uc=primitive_cell)
    for lvec in ([0,0],[-1,0],[0,-1]):
        h_model.add_hop(t, 0,1,lvec)
    for lvec in ([1,0],[-1,1],[0,-1]):
        h_model.add_hop(t2*np.exp(1.j*phi), 0,0,lvec)
    for lvec in ([-1,0],[1,-1],[0,1]):
        h_model.add_hop(t2*np.exp(1.j*phi), 1,1,lvec) 

    sc_model = h_model.supercell([L,L])
       
    return sc_model

def h_anderson_disorder_pythtb(model, w):
    n_orb = model.get_num_orbitals() 
    if w != 0.:
        for j in range (n_orb):
            dis = 0.5*w*(2*np.random.random()-1.0)
            model.set_onsite(dis, j, mode='add')

    return model

def h_anderson_disorder_tbmodels(model, w):
    n_orb = len(model.pos)
    if w != 0.:
        dis = 0.5*w*(2*np.random.rand(n_orb)-1.0)                   

        model.add_on_site(dis)

    return model