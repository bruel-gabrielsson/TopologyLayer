from __future__ import print_function

import torch

from torch.autograd import Variable, Function
from .cohom_cpp import SimplicialComplex, persistenceForward, persistenceBackward

class SubLevelSetDiagram(Function):
    """
    Compute sub-level set persistence on a space
    forward inputs:
        X - simplicial complex
        f - torch.float tensor of function values on vertices of X
        maxdim - maximum homology dimension
    """
    @staticmethod
    def forward(ctx, X, f, maxdim):
        f = f.view(-1,1)
        X.extendFloat(f)
        ret = persistenceForward(X, maxdim)
        ctx.X = X
        return tuple(ret)

    @staticmethod
    def backward(ctx, *grad_dgms):
        # print(grad_dgms)
        X = ctx.X
        grad_ret = list(grad_dgms)
        grad_f = persistenceBackward(X, grad_ret)
        return None, grad_f.view(-1,1), None
