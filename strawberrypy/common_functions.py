import numpy as np
import tbmodels as tbm
import pythtb as ptb

from . import _tbmodels
from . import _pythtb

######################################
# Finite model builder
######################################

def make_finite(model, nx_sites : int, ny_sites : int):
    """
    Make a finite mdoel along x and y direction by first cutting on the y direction and then on the x. This convention has been used to track the positions in the functions

        Args:
        - model : instance of the model, which should be periodic in both x and y direction
        - nx_sites, ny_sites : number of sites of the finite sample in both directions

        Returns:
        - model : the finite model
    """

    if isinstance(model, tbm.Model):
        return _tbmodels.make_finite(model, nx_sites, ny_sites)
    elif isinstance(model, ptb.tb_model):
        return _pythtb.make_finite(model, nx_sites, ny_sites)
    else:
        raise NotImplementedError("Invalid model instance")
    
######################################
# Misc utils
######################################

def commutator(A, B):
    return (A @ B - B @ A)

def partialtrace(matrix, num):
    if not int(len(matrix) / num) == len(matrix) / num:
        raise RuntimeError("Number of element to take the partial trace does not divide the matrix dimension.")
    return np.array([np.sum([matrix[i * num + k, i * num + k] for k in range(num)]) for i in range(int(len(matrix) / num))])

