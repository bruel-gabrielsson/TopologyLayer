import unittest

from topologylayer.functional.persistence import SimplicialComplex, persistenceForwardCohom
from topologylayer.util.process import remove_zero_bars
import torch
import numpy as np

class BasicLevelset(unittest.TestCase):
    def test(self):
        # first, we build our complex
        s = SimplicialComplex()

        # a cycle graph on vertices 1,2,3,4
        # cone with vertex 0
        s.append([0])
        s.append([1])
        s.append([2])
        s.append([3])
        s.append([4])

        s.append([0,1])
        s.append([0,2])
        s.append([0,3])
        s.append([0,4])

        s.append([1,2])
        s.append([1,3])
        s.append([4,2])
        s.append([4,3])

        s.append([0,1,2])
        s.append([0,1,3])
        s.append([0,2,4])
        s.append([0,3,4])

        # initialize internal data structures
        s.initialize()

        # function on vertices
        # we are doing sub-level set persistence
        # expect single H0 [0,inf]
        # expect single H1 [0,2]
        f = torch.Tensor([2., 0., 0., 0., 0.])

        # extend filtration to simplical complex
        s.extendFloat(f)

        # compute persistence with MAXDIM=1
        ret = persistenceForwardCohom(s, 1)

        self.assertEqual(
            torch.all(torch.eq(remove_zero_bars(ret[0]), torch.tensor([[0., np.inf]]))),
            True,
            "Unexpected 0-Dim persistence")
        self.assertEqual(
            torch.all(torch.eq(remove_zero_bars(ret[1]), torch.tensor([[0., 2.]]))),
            True,
            "Unexpected 1-Dim persistence")
