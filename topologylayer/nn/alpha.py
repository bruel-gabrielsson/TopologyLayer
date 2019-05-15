from ..functional.alpha import Diagramlayer as alphadgm
from ..util.process import remove_filler

import torch


class AlphaLayer(torch.nn.Module):
    """
    Alpha persistence layer for spatial inputs
    Should be equivalent for Rips, but much faster
    Parameters:
        maxdim : maximum homology dimension (defualt=0)
        verbose : print information (default=False)
    """

    def __init__(self, maxdim=0, verbose=False):
        super(AlphaLayer, self).__init__()
        self.verbose = verbose
        self.maxdim = maxdim
        self.fnobj = alphadgm()

    def forward(self, x):
        dgm = self.fnobj.apply(x, self.maxdim, self.verbose)
        dgms = tuple(remove_filler(dgm[i], -1) for i in range(self.maxdim+1))
        return dgms, True
