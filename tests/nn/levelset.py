import unittest

import topologylayer
import torch
import numpy as np
from topologylayer.util.process import remove_zero_bars

class Levelset1dsuper(unittest.TestCase):
    def test(self):
        from topologylayer.nn import LevelSetLayer1D

        # superlevel set
        layer = LevelSetLayer1D(size=3, sublevel=False)
        y = torch.tensor([1,0,1], dtype=torch.float).requires_grad_(True)

        dgms, issub = layer(y)
        self.assertEqual(
            issub,
            False,
            "Expected superlevel set layer")
        self.assertEqual(
            torch.all(torch.eq(remove_zero_bars(dgms[0]),
                        torch.tensor([[1., 0.], [1., -np.inf]]))),
            True,
            "unexpected barcode")

        p = torch.sum(dgms[0][1])
        p.backward()

        self.assertEqual(
            y.grad[1].item(),
            1.0,
            "unexpected gradient")


class Levelset1dsub(unittest.TestCase):
    def test(self):
        from topologylayer.nn import LevelSetLayer1D

        # sublevel set
        layer = LevelSetLayer1D(size=3, sublevel=True)
        y = torch.tensor([1,0,1], dtype=torch.float).requires_grad_(True)

        dgms, issub = layer(y)
        self.assertEqual(
            issub,
            True,
            "Expected sublevel set layer")
        self.assertEqual(
            torch.all(torch.eq(remove_zero_bars(dgms[0]),
                        torch.tensor([[0., np.inf]]))),
            True,
            "unexpected barcode")

        p = torch.sum(dgms[0][0])
        p.backward()

        self.assertEqual(
            y.grad[0].item(),
            2.0,
            "unexpected gradient")
