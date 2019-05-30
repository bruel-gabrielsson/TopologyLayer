from topologylayer.nn import AlphaLayer
from topologylayer.nn import *

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# folder for storing figures
figfolder = 'figures/'
import os
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

# seed for reproduciblity
np.random.seed(0)

# initial pointcloud
num_samples = 100
data = np.random.rand(num_samples, 2)
plt.figure(figsize=(5,5))
plt.scatter(data[:,0], data[:,1])
plt.axis('equal')
plt.axis('off')
plt.savefig(figfolder + 'uniform.png')

# alpha layer
layer = AlphaLayer(maxdim=1)

# Increase H1
print("Increase H1")
x = torch.autograd.Variable(torch.tensor(data).type(torch.float), requires_grad=True)
f1 = BarcodePolyFeature(1,2,0)
lr=1e-2
optimizer = torch.optim.Adam([x], lr=lr)
for i in range(100):
    optimizer.zero_grad()
    loss = -f1(layer(x))
    loss.backward()
    optimizer.step()
    if (i % 5 == 4):
        print ("[Iter %d] [top loss: %f]" % (i, loss.item()))

y = x.detach().numpy()
plt.figure(figsize=(5,5))
plt.scatter(y[:,0], y[:,1])
plt.axis('equal')
plt.axis('off')
plt.savefig(figfolder + 'uniform+H1.png')


# decrease H1
print("Decrease H1")
x = torch.autograd.Variable(torch.tensor(data).type(torch.float), requires_grad=True)
lr=1e-2
optimizer = torch.optim.Adam([x], lr=lr)
for i in range(100):
    optimizer.zero_grad()
    loss = f1(layer(x))
    loss.backward()
    optimizer.step()
    if (i % 5 == 4):
        print ("[Iter %d] [top loss: %f]" % (i, loss.item()))

y = x.detach().numpy()
plt.figure(figsize=(5,5))
plt.scatter(y[:,0], y[:,1])
plt.axis('equal')
plt.axis('off')
plt.savefig(figfolder + 'uniform-H1.png')


# decrease H0
print("Decrease H0")
x = torch.autograd.Variable(torch.tensor(data).type(torch.float), requires_grad=True)
layer = AlphaLayer(maxdim=0)
f2 = BarcodePolyFeature(0,2,0)
lr=1e-2
optimizer = torch.optim.Adam([x], lr=lr)
for i in range(100):
    optimizer.zero_grad()
    loss = f2(layer(x))
    loss.backward()
    optimizer.step()
    if (i % 5 == 4):
        print ("[Iter %d] [top loss: %f]" % (i, loss.item()))

y = x.detach().numpy()
plt.figure(figsize=(5,5))
plt.scatter(y[:,0], y[:,1])
plt.axis('equal')
plt.axis('off')
plt.savefig(figfolder + 'uniform-H0.png')


# increase H0
print("Increase H0")
x = torch.autograd.Variable(torch.tensor(data).type(torch.float), requires_grad=True)
lr=1e-2
optimizer = torch.optim.Adam([x], lr=lr)
for i in range(100):
    optimizer.zero_grad()
    loss = -f2(layer(x))
    loss.backward()
    optimizer.step()
    if (i % 5 == 4):
        print ("[Iter %d] [top loss: %f]" % (i, loss.item()))

y = x.detach().numpy()
plt.figure(figsize=(5,5))
plt.scatter(y[:,0], y[:,1])
plt.axis('equal')
plt.axis('off')
plt.savefig(figfolder + 'uniform+H0.png')


# Fun combination of H0, H1
print("Increase H1, decrease H0")
x = torch.autograd.Variable(torch.tensor(data).type(torch.float), requires_grad=True)
lr=1e-2
optimizer = torch.optim.Adam([x], lr=lr)
layer = AlphaLayer(maxdim=1)
f3 = BarcodePolyFeature(1,2,1)
f4 = BarcodePolyFeature(0,2,0)
for i in range(100):
    optimizer.zero_grad()
    dgminfo = layer(x)
    loss = -f3(dgminfo) + f4(dgminfo)
    loss.backward()
    optimizer.step()
    if (i % 5 == 4):
        print ("[Iter %d] [top loss: %f]" % (i, loss.item()))

y = x.detach().numpy()
plt.figure(figsize=(5,5))
plt.scatter(y[:,0], y[:,1])
plt.axis('equal')
plt.axis('off')
plt.savefig(figfolder + 'uniform+H1-H0.png')
