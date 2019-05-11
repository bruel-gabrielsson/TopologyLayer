from ..functional.alpha import Diagramlayer as alphadgm
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
        return dgm, True
