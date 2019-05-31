from ..functional.levelset_dionysus import Diagramlayer as levelsetdgm
from ..util.process import remove_filler

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import Delaunay
import dionysus as d

def init_freudenthal_2d(width, height):
    """
    Freudenthal triangulation of 2d grid
    """
    s = d.Filtration()
    # row-major format
    # 0-cells
    for i in range(height):
        for j in range(width):
            ind = i*width + j
            s.append(d.Simplex([ind]))
    # 1-cells
    for i in range(height):
        for j in range(width-1):
            ind = i*width + j
            s.append(d.Simplex([ind, ind + 1]))
    for i in range(height-1):
        for j in range(width):
            ind = i*width + j
            s.append(d.Simplex([ind, ind + width]))
    # 2-cells + diagonal 1-cells
    for i in range(height-1):
        for j in range(width-1):
            ind = i*width + j
            # diagonal
            s.append(d.Simplex([ind, ind + width + 1]))
            # 2-cells
            s.append(d.Simplex([ind, ind + 1, ind + width + 1]))
            s.append(d.Simplex([ind, ind + width, ind + width + 1]))
    return s


class LevelSetLayer(nn.Module):
    """
    Level set persistence layer
    Parameters:
        size : (width, height) - tuple for image input dimensions
        maxdim : haximum homology dimension (default 1)
        complex :
            "scipy" - use scipy freudenthal triangulation (default)
            "freudenthal" - use canonical freudenthal triangulation
    """
    def __init__(self, size, maxdim=1, complex="scipy"):
        super(LevelSetLayer, self).__init__()
        self.size = size
        self.maxdim = maxdim
        self.fnobj = levelsetdgm()

        # extract width and height
        width, height = size
        if complex == "scipy":
            # initialize complex to use for persistence calculations
            axis_x = np.arange(0, width)
            axis_y = np.arange(0, height)
            grid_axes = np.array(np.meshgrid(axis_x, axis_y))
            grid_axes = np.transpose(grid_axes, (1, 2, 0))

            # creation of a complex for calculations
            tri = Delaunay(grid_axes.reshape([-1, 2]))
            faces = tri.simplices.copy()
            self.complex = self.fnobj.init_filtration(faces)
        elif complex == "freudenthal":
            self.complex = init_freudenthal_2d(width, height)
        else:
            AssertionError("bad complex type")

    def forward(self, img):
        dgm = self.fnobj.apply(img, self.complex)
        #dgm = dgm[0:(self.maxdim+1),:,:]
        dgms = tuple(remove_filler(dgm[i], -np.inf) for i in range(self.maxdim+1))
        return dgms, False


def init_line_complex(p):
    """
    initialize 1D complex on the line
    Input:
        p - number of 0-simplices
    Will add (p-1) 1-simplices
    """
    f = d.Filtration()
    for i in range(p-1):
        c = d.closure([d.Simplex([i, i+1])], 1)
        for j in c:
            f.append(j)
    return f


class LevelSetLayer1D(nn.Module):
    """
    Level set persistence layer
    Parameters:
        size : number of features
    only returns H0
    """
    def __init__(self, size):
        super(LevelSetLayer1D, self).__init__()
        self.size = size
        self.fnobj = levelsetdgm()
        self.complex = init_line_complex(size)

    def forward(self, img):
        dgm = self.fnobj.apply(img, self.complex)
        dgm = dgm[0]
        return (dgm,), False
