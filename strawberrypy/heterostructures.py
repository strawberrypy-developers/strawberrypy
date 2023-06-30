import tbmodels as tbm
import pythtb as ptb
from copy import deepcopy

from .package import FiniteModel
from . import _tbmodels
from . import _pythtb

def make_heterostructure(model1 : FiniteModel, model2 : FiniteModel, direction : int, start : int, stop : int):
    """
    Modify a finite model by merging another system in it. The system will be split in the direction starting from start.

        Args:
        - model1, model2: two FiniteModel which composes the heterostructure
        - direction : direction in which the splitting happen, allowed 0 for 'x' or 1 for 'y'
        - start : starting point for the splitting in the 'direction' direction
        - end : end point of the splitting in the 'direction' direction

        Returns:
        - model : the a FiniteModel composed my the two subsystems
    """
    if model1._disordered:
        return_instance = deepcopy(model1)
    else:
        return_isntance = deepcopy(model2)
    
    if isinstance(model1._model, tbm.Model) and isinstance(model2._model, tbm.Model):
        return_instance._model = _tbmodels.make_heterostructure(model1._model, model2._model, model1._nx_sites, model1._ny_sites, direction, start, stop)
    elif isinstance(model1._model, ptb.tb_model) and isinstance(model2._model, ptb.tb_model):
        return_instance._model = _pythtb.make_heterostructure(model1._model, model2._model, model1._nx_sites, model1._ny_sites, direction, start, stop)
    else:
        raise NotImplementedError('Invalid model.')
    
    return return_instance