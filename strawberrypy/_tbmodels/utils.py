import numpy as np
import tbmodels as tbm

def _uc_vol(model):
    return np.linalg.norm(np.cross(model.uc[0], model.uc[1]))