from __future__ import print_function
import unittest

import topologylayer
import torch
import numpy as np
from topologylayer.util.process import remove_zero_bars, remove_infinite_bars

class Levelset1dsuper(unittest.TestCase):
    def test(self):
        from topologylayer.nn import LevelSetLayer1D

        # superlevel set
        for alg in ['hom', 'hom2', 'cohom']:
            layer = LevelSetLayer1D(size=3, sublevel=False, alg=alg)
            y = torch.tensor([1,0,1], dtype=torch.float).requires_grad_(True)

            dgms, issub = layer(y)
            self.assertEqual(
                issub,
                False,
                "Expected superlevel set layer. alg=" + alg)
            self.assertEqual(
                torch.all(torch.eq(remove_infinite_bars(remove_zero_bars(dgms[0]), issub),
                            torch.tensor([[1., 0.]]))),
                True,
                "unexpected barcode. alg=" + alg)

            p = torch.sum(remove_infinite_bars(remove_zero_bars(dgms[0]), issub)[0])
            p.backward()

            self.assertEqual(
                y.grad[1].item(),
                1.0,
                "unexpected gradient. alg=" + alg)


class Levelset1dsub(unittest.TestCase):
    def test(self):
        from topologylayer.nn import LevelSetLayer1D

        # sublevel set
        for alg in ['hom', 'hom2', 'cohom']:
            layer = LevelSetLayer1D(size=3, sublevel=True, alg=alg)
            y = torch.tensor([1,0,1], dtype=torch.float).requires_grad_(True)

            dgms, issub = layer(y)
            self.assertEqual(
                issub,
                True,
                "Expected sublevel set layer. alg=" + alg)
            self.assertEqual(
                torch.all(torch.eq(remove_zero_bars(dgms[0]),
                            torch.tensor([[0., np.inf]]))),
                True,
                "unexpected barcode. alg=" + alg)

        # p = torch.sum(dgms[0][0])
        # p.backward()
        #
        # self.assertEqual(
        #     y.grad[0].item(),
        #     2.0,
        #     "unexpected gradient")


class Levelset2dsuper(unittest.TestCase):
    def test(self):
        from topologylayer.nn import LevelSetLayer2D

        # sublevel set
        for alg in ['hom', 'hom2', 'cohom']:
            layer = LevelSetLayer2D(size=(3,3), maxdim=1, sublevel=False, alg=alg)
            x = torch.tensor([[2, 1, 1],[1, 0.5, 1],[1, 1, 1]], dtype=torch.float).requires_grad_(True)

            dgms, issub = layer(x)
            self.assertEqual(
                issub,
                False,
                "Expected superlevel set layer. alg=" + alg)
            self.assertEqual(
                torch.all(torch.eq(remove_zero_bars(dgms[0]),
                            torch.tensor([[2., -np.inf]]))),
                True,
                "unexpected 0-dim barcode. alg=" + alg)
            self.assertEqual(
                torch.all(torch.eq(remove_zero_bars(dgms[1]),
                            torch.tensor([[1., 0.5]]))),
                True,
                "unexpected 1-dim barcode. alg=" + alg)

            d1 = remove_zero_bars(dgms[1])
            p = d1[0,0] - d1[0,1]
            p.backward()

            self.assertEqual(
                x.grad[1,1].item(),
                -1.0,
                "unexpected gradient. alg=" + alg)
