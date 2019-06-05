from __future__ import print_function

import torch

from torch.autograd import Variable, Function
from .cohom_cpp import SimplicialComplex, persistenceForward, persistenceBackward, persistenceForwardHom

class SubLevelSetDiagram(Function):
    """
    Compute sub-level set persistence on a space
    forward inputs:
        X - simplicial complex
        f - torch.float tensor of function values on vertices of X
        maxdim - maximum homology dimension
        alg - algorithm
            'hom' = homology (default)
            'cohom' = cohomology
    """
    @staticmethod
    def forward(ctx, X, f, maxdim, alg='hom'):
        ctx.retshape = f.shape
        f = f.view(-1)
        X.extendFloat(f)
        if alg == 'hom':
            ret = persistenceForwardHom(X, maxdim)
        elif alg == 'cohom':
            ret = persistenceForward(X, maxdim)
        ctx.X = X
        return tuple(ret)

    @staticmethod
    def backward(ctx, *grad_dgms):
        # print(grad_dgms)
        X = ctx.X
        retshape = ctx.retshape
        grad_ret = list(grad_dgms)
        grad_f = persistenceBackward(X, grad_ret)
        return None, grad_f.view(retshape), None, None
