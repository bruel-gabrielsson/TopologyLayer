from __future__ import print_function
from topologylayer.functional.cohom_cpp import SimplicialComplex, persistenceForward
from topologylayer.util.process import remove_zero_bars
import torch


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
ret = persistenceForward(s, 1)

for k in range(2):
    print("dimension %d bars" % k)
    print(remove_zero_bars(ret[k]))
