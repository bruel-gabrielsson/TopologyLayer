from __future__ import print_function
from topologylayer.nn.levelset_cpp import LevelSetLayer1D as LevelSetLayer1Dnew
from topologylayer.nn.levelset_cpp import LevelSetLayer2D as LevelSetLayer2Dnew
from topologylayer.nn.levelset import LevelSetLayer1D as LevelSetLayer1Dold
from topologylayer.nn.levelset import LevelSetLayer as LevelSetLayer2Dold

import torch
import time
import numpy as np


def sum_finite(d):
    diff = d[:,0] - d[:,1]
    inds = diff < np.inf
    return torch.sum(diff[inds])


# apparently there is some overhead the first time backward is called.
# we'll just get it over with now.
n = 20
y = torch.rand(n, 1, dtype=torch.float).requires_grad_(True)
layer1 = LevelSetLayer1Dnew(n, False)
dgm, issublevel = layer1(y)
p = sum_finite(dgm[0])
p.backward()

for n in [100, 200, 400, 1000, 2000]:
    y = torch.rand(n, 1, dtype=torch.float).requires_grad_(True)
    print("\n\n1D complexes n = %d" % n)

    t0 = time.time()
    layer1 = LevelSetLayer1Dnew(n, False)
    ta = time.time() - t0
    print("\nnew construction = %f sec" % ta)
    t0 = time.time()
    layer2 = LevelSetLayer1Dold(n)
    tb = time.time() - t0
    print("old construction = %f sec" % tb)
    print("factor improvement = %f" % (tb/ta))

    t0 = time.time()
    dgm, issublevel = layer1(y)
    ta = time.time() - t0
    print("\nnew forward = %f sec" % ta)
    t0 = time.time()
    dgm2, issublevel2 = layer2(y)
    tb = time.time() - t0
    print("old forward = %f sec" % tb)
    print("factor improvement = %f" % (tb/ta))

    p = sum_finite(dgm[0])
    t0 = time.time()
    p.backward()
    ta = time.time() - t0
    print("\nnew backward = %f sec" % ta)
    p = sum_finite(dgm2[0])
    t0 = time.time()
    p.backward()
    tb = time.time() - t0
    print("old backward = %f sec" % tb)
    print("factor improvement = %f" % (tb/ta))

for size in [(28,28), (64,64), (128,128)]:
    x = torch.rand(*size, dtype=torch.float).requires_grad_(True)
    print("\n\n2D complexes size =", size)

    t0 = time.time()
    layer1 = LevelSetLayer2Dnew(size, sublevel=False)
    ta = time.time() - t0
    print("\nnew construction = %f sec" % ta)
    t0 = time.time()
    layer2 = LevelSetLayer2Dold(size)
    tb = time.time() - t0
    print("old construction = %f sec" % tb)
    print("factor improvement = %f" % (tb/ta))

    t0 = time.time()
    dgm, issublevel = layer1(x.view(-1,1))
    ta = time.time() - t0
    print("\nnew forward = %f sec" % ta)
    t0 = time.time()
    dgm2, issublevel2 = layer2(x.view(-1,1))
    tb = time.time() - t0
    print("old forward  = %f sec" % tb)
    print("factor improvement = %f" % (tb/ta))

    p = sum_finite(dgm[0])
    t0 = time.time()
    p.backward()
    ta = time.time() - t0
    print("\nnew backward = %f sec" % ta)
    p = sum_finite(dgm2[0])
    t0 = time.time()
    p.backward()
    tb = time.time() - t0
    print("old backward = %f sec" % tb)
    print("factor improvement = %f" % (tb/ta))
# print dgm
# print dgm2
