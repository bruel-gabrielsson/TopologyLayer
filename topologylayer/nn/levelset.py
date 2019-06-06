from ..functional.sublevel import SubLevelSetDiagram
from topologylayer.functional.persistence import SimplicialComplex
from topologylayer.util.construction import unique_simplices

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import Delaunay
import itertools


class LevelSetLayer(nn.Module):
    """
    Level set persistence layer arbitrary simplicial complex
    Parameters:
        complex : SimplicialComplex
        maxdim : maximum homology dimension (default 1)
        sublevel : sub or superlevel persistence (default=True)
        alg : algorithm
            'hom' = homology (default)
            'cohom' = cohomology

    Note that the complex should be acyclic for the computation to be correct (currently)
    """
    def __init__(self, complex, maxdim=1, sublevel=True, alg='hom'):
        super(LevelSetLayer, self).__init__()
        self.complex = complex
        self.maxdim = maxdim
        self.fnobj = SubLevelSetDiagram()
        self.sublevel = sublevel
        self.alg = alg

        # make sure complex is initialized
        self.complex.initialize()


    def forward(self, f):
        if self.sublevel:
            dgms = self.fnobj.apply(self.complex, f, self.maxdim, self.alg)
            return dgms, True
        else:
            f = -f
            dgms = self.fnobj.apply(self.complex, f, self.maxdim, self.alg)
            dgms = tuple(-dgm for dgm in dgms)
            return dgms, False



def init_tri_complex(width, height):
    """
    initialize 2d complex in dumbest possible way
    """
    # initialize complex to use for persistence calculations
    axis_x = np.arange(0, width)
    axis_y = np.arange(0, height)
    grid_axes = np.array(np.meshgrid(axis_x, axis_y))
    grid_axes = np.transpose(grid_axes, (1, 2, 0))

    # creation of a complex for calculations
    tri = Delaunay(grid_axes.reshape([-1, 2]))
    return unique_simplices(tri.simplices, 2)


def init_freudenthal_2d(width, height):
    """
    Freudenthal triangulation of 2d grid
    """
    s = SimplicialComplex()
    # row-major format
    # 0-cells
    for i in range(height):
        for j in range(width):
            ind = i*width + j
            s.append([ind])
    # 1-cells
    for i in range(height):
        for j in range(width-1):
            ind = i*width + j
            s.append([ind, ind + 1])
    for i in range(height-1):
        for j in range(width):
            ind = i*width + j
            s.append([ind, ind + width])
    # 2-cells + diagonal 1-cells
    for i in range(height-1):
        for j in range(width-1):
            ind = i*width + j
            # diagonal
            s.append([ind, ind + width + 1])
            # 2-cells
            s.append([ind, ind + 1, ind + width + 1])
            s.append([ind, ind + width, ind + width + 1])
    return s


def init_grid_2d(width, height):
    """
    initialize 2d grid with diagonal and anti-diagonal
    """
    s = SimplicialComplex()
    # row-major format
    # 0-cells
    for i in range(height):
        for j in range(width):
            ind = i*width + j
            s.append([ind])
    # 1-cells
    for i in range(height):
        for j in range(width-1):
            ind = i*width + j
            s.append([ind, ind + 1])
    for i in range(height-1):
        for j in range(width):
            ind = i*width + j
            s.append([ind, ind + width])
    # 2-cells + diagonal 1-cells
    for i in range(height-1):
        for j in range(width-1):
            ind = i*width + j
            # diagonal
            s.append([ind, ind + width + 1])
            # 2-cells
            s.append([ind, ind + 1, ind + width + 1])
            s.append([ind, ind + width, ind + width + 1])
    # 2-cells + anti-diagonal 1-cells
    for i in range(height-1):
        for j in range(width-1):
            ind = i*width + j
            # anti-diagonal
            s.append([ind + 1, ind + width])
            # 2-cells
            s.append([ind + 1, ind + width, ind + width + 1])
            s.append([ind, ind + 1, ind + width])
    return s


class LevelSetLayer2D(LevelSetLayer):
    """
    Level set persistence layer for 2D input
    Parameters:
        size : (width, height) - tuple for image input dimensions
        maxdim : maximum homology dimension (default 1)
        sublevel : sub or superlevel persistence (default=True)
        complex : method of constructing complex
            "freudenthal" (default) - canonical triangulation of the lattice
            "grid" - includes diagonals and anti-diagonals
            "delaunay" - scipy delaunay triangulation of the lattice.
                Every square will be triangulated, but the diagonal orientation may not be consistent.
        alg : algorithm
            'hom' = homology (default)
            'cohom' = cohomology
    """
    def __init__(self, size, maxdim=1, sublevel=True, complex="freudenthal", alg='hom'):
        width, height = size
        tmpcomplex = None
        if complex == "freudenthal":
            tmpcomplex = init_freudenthal_2d(width, height)
        elif complex == "grid":
            tmpcomplex = init_grid_2d(width, height)
        elif complex == "delaunay":
            tmpcomplex = init_tri_complex(width, height)
        super(LevelSetLayer2D, self).__init__(tmpcomplex, maxdim=maxdim, sublevel=sublevel, alg=alg)
        self.size = size


def init_line_complex(p):
    """
    initialize 1D complex on the line
    Input:
        p - number of 0-simplices
    Will add (p-1) 1-simplices
    """
    s = SimplicialComplex()
    for i in range(p):
        s.append([i])
    for i in range(p-1):
        s.append([i, i+1])
    return s


class LevelSetLayer1D(LevelSetLayer):
    """
    Level set persistence layer
    Parameters:
        size : number of features
        sublevel : True=sublevel persistence, False=superlevel persistence
        alg : algorithm
            'hom' = homology (default)
            'cohom' = cohomology
    only returns H0
    """
    def __init__(self, size, sublevel=True, alg='hom'):
        super(LevelSetLayer1D, self).__init__(
            init_line_complex(size),
            maxdim=0,
            sublevel=sublevel,
            alg=alg
            )
