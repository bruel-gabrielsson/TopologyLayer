from __future__ import print_function
from topologylayer.nn import LevelSetLayer1D
import matplotlib.pyplot as plt

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
y = torch.rand(n, dtype=torch.float).requires_grad_(True)
layer1 = LevelSetLayer1D(n, False)
dgm, issublevel = layer1(y)
p = sum_finite(dgm[0])
p.backward()

algs = ['hom', 'cohom']

tcs = {}
tfs = {}
tbs = {}
for alg in algs:
    tcs[alg] = []
    tfs[alg] = []
    tbs[alg] = []


ns = [100, 200, 400, 1000, 2000, 4000, 8000, 16000]

for alg in algs:
    for n in ns:
        y = torch.rand(n, dtype=torch.float).requires_grad_(True)

        t0 = time.time()
        layer = LevelSetLayer1D(n, False, alg=alg)
        ta = time.time() - t0
        tcs[alg].append(ta)

        t0 = time.time()
        dgm, issublevel = layer(y)
        ta = time.time() - t0
        tfs[alg].append(ta)

        p = sum_finite(dgm[0])
        t0 = time.time()
        p.backward()
        ta = time.time() - t0
        tbs[alg].append(ta)

for alg in algs:
    plt.loglog(ns, tfs[alg], label=alg)
plt.legend()
plt.xlabel("n")
plt.ylabel("forward time")
plt.savefig("alg_time_forward.png")
