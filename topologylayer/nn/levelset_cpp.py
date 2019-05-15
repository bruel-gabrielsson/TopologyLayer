from ..functional.sublevel import SubLevelSetDiagram
from ..functional.cohom_cpp import SimplicialComplex

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import Delaunay
import itertools

import itertools

# function to get unique values
# this is disgusting and slow
def unique(list0):

    # intilize a null list
    unique_list = []
    # traverse for all elements
    for x in list0:
        # check if exists in unique_list or not
        in_list = False
        for y in unique_list:
            if np.array_equal(x, y):
                in_list = True
                break
        if not in_list:
            unique_list.append(x)
    return unique_list

def init_tri_complex(faces, maxdim):
    simplices = []
    # fill in higher-d - cells
    for s in faces:
        # simplices
        for dim in range(0, maxdim+1):
            # loop over faces
            for face in itertools.combinations(s, dim+1):
                simplices.append(np.sort(list(face)))
    simplices = unique(simplices)
    s = SimplicialComplex()
    for c in simplices:
        s.append(list(c))
    return s


class LevelSetLayer(nn.Module):
    """
    Level set persistence layer for 2D input
    Parameters:
        size : (width, height) - tuple for image input dimensions
        maxdim : maximum homology dimension (default 1)
        sublevel : sub or superlevel persistence (default=True)
    """
    def __init__(self, size, maxdim=1, sublevel=True):
        super(LevelSetLayer, self).__init__()
        self.size = size
        self.maxdim = maxdim
        self.fnobj = SubLevelSetDiagram()
        self.sublevel = sublevel

        # extract width and height
        width, height = size
        # initialize complex to use for persistence calculations
        axis_x = np.arange(0, width)
        axis_y = np.arange(0, height)
        grid_axes = np.array(np.meshgrid(axis_x, axis_y))
        grid_axes = np.transpose(grid_axes, (1, 2, 0))

        # creation of a complex for calculations
        tri = Delaunay(grid_axes.reshape([-1, 2]))
        faces = tri.simplices.copy()
        self.complex = init_tri_complex(faces, self.maxdim+1)
        self.complex.initialize()


    def forward(self, f):
        if self.sublevel:
            dgms = self.fnobj.apply(self.complex, f, self.maxdim)
            return dgms, True
        else:
            f = -f
            dgms = self.fnobj.apply(self.complex, f, self.maxdim)
            dgms = tuple(-dgm for dgm in dgms)
            return dgms, False



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


class LevelSetLayer1D(nn.Module):
    """
    Level set persistence layer
    Parameters:
        size : number of features
        sublevel : True=sublevel persistence, False=superlevel persistence
    only returns H0
    """
    def __init__(self, size, sublevel=True):
        super(LevelSetLayer1D, self).__init__()
        self.size = size
        self.fnobj = SubLevelSetDiagram()
        self.complex = init_line_complex(size)
        self.complex.initialize()
        self.sublevel = sublevel

    def forward(self, f):
        if self.sublevel:
            dgm, = self.fnobj.apply(self.complex, f, 0) # only 0 dim homology
            return dgm, True
        else:
            f = -f
            dgm, = self.fnobj.apply(self.complex, f, 0) # only 0 dim homology
            dgm = -dgm
            return dgm, False
