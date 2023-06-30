import pythtb as ptb
import numpy as np

def _uc_vol(model):
    return np.linalg.norm(np.cross(model._lat[0], model._lat[1]))