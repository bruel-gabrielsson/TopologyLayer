from __future__ import print_function

from torch.autograd import Variable, Function
from .persistence import SimplicialComplex, persistenceForwardCohom, persistenceBackwardFlag, persistenceForwardHom

class FlagDiagram(Function):
    """
    Compute Flag complex persistence using point coordinates

    forward inputs:
        X - simplicial complex
        y - N x D torch.float tensor of coordinates
        maxdim - maximum homology dimension
        alg - algorithm
            'hom' = homology (default)
            'hom2' = nz suppressing homology variant
            'cohom' = cohomology
    """
    @staticmethod
    def forward(ctx, X, y, maxdim, alg='hom'):
        device = y.device
        ctx.device = device
        ycpu = y.cpu()
        X.extendFlag(ycpu)
        if alg == 'hom':
            ret = persistenceForwardHom(X, maxdim, 0)
        elif alg == 'hom2':
            ret = persistenceForwardHom(X, maxdim, 1)
        elif alg == 'cohom':
            ret = persistenceForwardCohom(X, maxdim)
        ctx.X = X
        ctx.save_for_backward(ycpu)
        ret = [r.to(device) for r in ret]
        return tuple(ret)

    @staticmethod
    def backward(ctx, *grad_dgms):
        # print(grad_dgms)
        X = ctx.X
        device = ctx.device
        ycpu, = ctx.saved_tensors
        grad_ret = [gd.cpu() for gd in grad_dgms]
        grad_y = persistenceBackwardFlag(X, ycpu, grad_ret)
        return None, grad_y.to(device), None, None
