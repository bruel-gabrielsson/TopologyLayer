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
        ret = persistenceForward(X, f, maxdim)
        dgms = [x[0] for x in ret]
        inds = [x[1] for x in ret]
        ctx.X = X
        ctx.inds = inds
        return tuple(dgms)

    @staticmethod
    def backward(ctx, *grad_dgms):
        X = ctx.X
        inds = ctx.inds
        grad_ret = [[grad_dgms[i], inds[i]] for i in range(len(inds))]
        grad_f = persistenceBackward(X, grad_ret)
        return None, grad_f, None
