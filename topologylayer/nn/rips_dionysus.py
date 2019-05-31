from ..functional.levelset_dionysus import Diagramlayer as ripsdgm
from ..util.process import remove_filler

import torch
import torch.nn as nn
import numpy as np

class RipsLayer(nn.Module):
    """
    Rips persistence layer
    Parameters:
        maxdim : maximum homology dimension (default=1)
        rmax   : maximum value of filtration (default=inf)
        verbose : print information
    """
    def __init__(self, maxdim=1, rmax=np.inf, verbose=False):
        super(RipsLayer, self).__init__()
        self.rmax = rmax
        self.maxdim = maxdim
        self.verbose = verbose
        self.fnobj = ripsdgm()

    def forward(self, x):
        dgm = self.fnobj.apply(x, self.rmax, self.maxdim, self.verbose)
        dgms = tuple(remove_filler(dgm[i], -np.inf) for i in range(self.maxdim+1))
        return dgms, True
