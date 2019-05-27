from __future__ import print_function

from torch.autograd import Variable, Function
from .cohom_cpp import SimplicialComplex, persistenceForward, persistenceBackwardFlag

class FlagDiagram(Function):
    """
    Compute Flag complex persistence using point coordinates

    forward inputs:
        X - simplicial complex
        y - N x D torch.float tensor of coordinates
        maxdim - maximum homology dimension
    """
    @staticmethod
    def forward(ctx, X, y, maxdim):
        X.extendFlag(y)
        ret = persistenceForward(X, maxdim)
        ctx.X = X
        ctx.save_for_backward(y)
        return tuple(ret)

    @staticmethod
    def backward(ctx, *grad_dgms):
        # print(grad_dgms)
        X = ctx.X
        y, = ctx.saved_variables
        grad_ret = list(grad_dgms)
        grad_y = persistenceBackwardFlag(X, y, grad_ret)
        return None, grad_y, None
