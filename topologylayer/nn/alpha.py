from topologylayer.util.construction import unique_simplices
from topologylayer.functional.flag import FlagDiagram
from topologylayer.functional.persistence import SimplicialComplex
from scipy.spatial import Delaunay

import torch
import torch.nn as nn
import numpy as np


def delaunay_complex_1d(x):
    """
    returns Delaunay complex on 1D space
    """
    inds = np.argsort(x)
    s = SimplicialComplex()
    # append cells in sorted order
    s.append([inds[0]])
    for ii in range(len(inds) - 1):
        s.append([inds[ii+1]])
        s.append([inds[ii], inds[ii+1]])
    return s


def delaunay_complex(x, maxdim=2):
    """
    compute Delaunay triangulation
    fill in simplices as appropriate

    if x is 1-dimensional, defaults to 1D Delaunay
    inputs:
        x - pointcloud
        maxdim - maximal simplex dimension (default = 2)
    """
    if x.shape[1] == 1:
        x = x.flatten()
        return alpha_complex_1d(x)
    tri = Delaunay(x)
    return unique_simplices(tri.simplices, maxdim)


class AlphaLayer(nn.Module):
    """
    Alpha persistence layer
    Parameters:
        maxdim : maximum homology dimension (default=0)
        alg : algorithm
            'hom' = homology (default)
            'cohom' = cohomology
    """
    def __init__(self, maxdim=0, alg='hom'):
        super(AlphaLayer, self).__init__()
        self.maxdim = maxdim
        self.fnobj = FlagDiagram()
        self.alg = alg

    def forward(self, x):
        xnp = x.cpu().detach().numpy()
        complex = None
        if xnp.shape[1] == 1:
            xnp = xnp.flatten()
            complex = delaunay_complex_1d(xnp)
        else:
            complex = delaunay_complex(xnp, maxdim=self.maxdim+1)
        complex.initialize()
        dgms = self.fnobj.apply(complex, x, self.maxdim, self.alg)
        return dgms, True
