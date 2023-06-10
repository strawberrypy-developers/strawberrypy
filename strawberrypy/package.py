from . import _pythtb
from . import _tbmodels

import pythtb 
import tbmodels

def single_point_chern(model, formula='symmetric'):
    if isinstance(model, pythtb.tb_model):
        return _pythtb.single_point_chern(model, formula)
    elif isinstance(model, tbmodels.Model):
        return _tbmodels.single_point_chern(model, formula)
    else:
        raise NotImplementedError('Invalid model.')

def single_point_spin_chern(model, spin='down', formula='symmetric', return_gap_pszp=False):
    if isinstance(model, pythtb.tb_model):
        return _pythtb.single_point_spin_chern(model, spin, formula, return_gap_pszp)
    elif isinstance(model, tbmodels.Model):
        return _tbmodels.single_point_spin_chern(model, spin, formula, return_gap_pszp)
    else:
        raise NotImplementedError('Invalid model.')
    
# Various lattice functions
def onsite_disorder(model, w : float, spinstates : int = 2, seed : int = None):
    """
    Add an Anderson onsite term in the hamiltonian, randomly distributed in [-w/2, w/2].

        Args:
        - model : the model to modify
        - w : maximum amplitude of disorder
        - spinstates : spin of the model
        - seed : seed for random number generator

        Returns:
        - model : the model with added onsite disorder
    """
    if isinstance(model, tbmodels.Model):
        return _tbmodels.onsite_disorder(model, w, spinstates, seed)
    elif isinstance(model, pythtb.tb_model):
        return _pythtb.onsite_disorder(model, w, spinstates, seed)
    else:
        raise NotImplementedError("Invalid model.")
    
# Building finite systems
    
def make_finite(model, nx_sites : int = 1, ny_sites : int = 1):
    """
    Make a finite mdoel along x and y direction by first cutting on the y direction and then on the x. This convention has been used to track the positions in the functions

        Args:
        - model : instance of the model, which should be periodic in both x and y direction
        - nx_sites, ny_sites : number of sites of the finite sample in both directions

        Returns:
        - model : the finite model
    """
    if isinstance(model, tbmodels.Model):
        return _tbmodels.make_finite(model, nx_sites, ny_sites)
    elif isinstance(model, pythtb.tb_model):
        return _pythtb.make_finite(model, nx_sites, ny_sites)
    else:
        raise NotImplementedError("Invalid model.")
    
def make_heterostructure(model1 , model2,  nx_sites : int, ny_sites : int, direction : int, start : int, stop : int):
    """
    Modify a finite model by merging another system in it. The system will be split in the direction starting from start.

        Args:
        - model1, model2: the models which composes the heterostructure
        - nx_sites, ny_sites : x and y length of the finite system
        - direction : direction in which the splitting happen, allowed 0 for 'x' or 1 for 'y'
        - start : starting point for the splitting in the 'direction' direction
        - end : end point of the splitting in the 'direction' direction

        Returns:
        - model : the model composed my the two subsystems
    """
    if isinstance(model1, tbmodels.Model) and isinstance(model2, tbmodels.Model):
        return _tbmodels.make_heterostructure(model1 , model2,  nx_sites, ny_sites, direction, start, stop)
    elif isinstance(model1, pythtb.tb_model) and isinstance(model2, pythtb.tb_model):
        return _pythtb.make_heterostructure(model1 , model2,  nx_sites, ny_sites, direction, start, stop)
    else:
        raise NotImplementedError('Invalid model.')

# Local marker functions

def local_chern_marker(model, nx_sites : int, ny_sites : int, direction : int = None, start : int = 0, return_projector : bool = None, projector = None, macroscopic_average : bool = False, cutoff : float = 0.8):
    """
    Evaluate the Z Chern marker on the whole lattice if direction is None. If direction is not None evaluates the Z Chern marker along direction starting from start.
    
        Args:
        - model : instance of the model
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

    if isinstance(model, tbmodels.Model):
        return _tbmodels.local_chern_marker(model, nx_sites, ny_sites, direction, start, return_projector, projector, macroscopic_average, cutoff)
    elif isinstance(model, pythtb.tb_model):
        return _pythtb.local_chern_marker(model, nx_sites, ny_sites, direction, start, return_projector, projector, macroscopic_average, cutoff)
    else:
        raise NotImplementedError('Invalid model.')