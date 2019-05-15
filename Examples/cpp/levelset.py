from topologylayer.nn.levelset_cpp import LevelSetLayer1D as LevelSetLayer1Dnew
from topologylayer.nn.levelset import LevelSetLayer1D as LevelSetLayer1Dold

import torch
import time

n = 100
y = torch.rand(n, dtype=torch.float)

t0 = time.time()
layer1 = LevelSetLayer1Dnew(n, False)
print time.time() - t0
t0 = time.time()
layer2 = LevelSetLayer1Dold(n)
print time.time() - t0

t0 = time.time()
dgm, issublevel = layer1(y)
print time.time() - t0
t0 = time.time()
dgm2, issublevel2 = layer2(y)
print time.time() - t0

# print dgm
# print dgm2
