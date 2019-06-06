from topologylayer.util.construction import clique_complex
from topologylayer.functional.flag import FlagDiagram

import torch
import torch.nn as nn
import numpy as np

class RipsLayer(nn.Module):
    """
    Rips persistence layer
    Parameters:
        n : number of points
        maxdim : maximum homology dimension (default=1)
        alg : algorithm
            'hom' = homology (default)
            'cohom' = cohomology
    """
    def __init__(self, n, maxdim=1, alg='hom'):
        super(RipsLayer, self).__init__()
        self.maxdim = maxdim
        self.complex = clique_complex(n, maxdim+1)
        self.complex.initialize()
        self.fnobj = FlagDiagram()
        self.alg = alg

    def forward(self, x):
        dgms = self.fnobj.apply(self.complex, x, self.maxdim, self.alg)
        return dgms, True
