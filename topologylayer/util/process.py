# processing functions for diagrams

import torch
import numpy as np

def remove_filler(dgm, val=np.inf):
    """
    remove filler rows from diagram
    """
    inds = (dgm[:,0] != val)
    return dgm[inds,:]


def remove_zero_bars(dgm):
    """
    remove zero bars from diagram
    """
    inds = dgm[:,0] != dgm[:,1]
    return dgm[inds,:]
