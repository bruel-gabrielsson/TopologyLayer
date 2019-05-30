import unittest

import topologylayer
import torch
import numpy as np
from topologylayer.util.process import remove_zero_bars

class RipsTest(unittest.TestCase):
    def test(self):
        from topologylayer.nn import RipsLayer

        # superlevel set
        layer = RipsLayer(4, maxdim=1)
        x = torch.tensor([[1, 1], [1,-1], [-1,-1], [-1,1]], dtype=torch.float).requires_grad_(True)

        dgms, issub = layer(x)
        self.assertEqual(
            issub,
            True,
            "Expected sublevel set layer")
        self.assertEqual(
            torch.all(torch.eq(remove_zero_bars(dgms[0]),
                        torch.tensor([[0., 2.], [0., 2.], [0., 2.], [0., np.inf]]))),
            True,
            "unexpected 0-dim barcode")
        self.assertEqual(
            torch.all(torch.eq(remove_zero_bars(dgms[1]),
                        torch.tensor([[2., 2.8284270763397217]]))),
            True,
            "unexpected 1-dim barcode")

        d0 = remove_zero_bars(dgms[0])
        p = d0[0, 1] - d0[0, 0]
        p.backward()

        self.assertEqual(
            torch.all(torch.eq(x.grad,
                        torch.tensor([[0,1],[0,-1],[0,0],[0,0]], dtype=torch.float))),
            True,
            "unexpected gradient")
