import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision.utils import save_image

import sys
sys.path.append('../Python')
sys.path.append('../Pytorch')

import matplotlib.pyplot as plt
import numpy as np
import time
from pprint import pprint
import os
from top_utils import *
from DiagramlayerRips import Diagramlayer as DiagramlayerRips
from DiagramlayerTopLevel import Diagramlayer as DiagramlayerToplevel

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

''''''
USE_GPU = True
''''''

dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    map_location = 'cuda'
    device = torch.device(map_location)
else:
    map_location = 'cpu'
    device = torch.device('cpu')
print('using device:', device)

ape = (1, 28, 28)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(ape))),
            nn.Tanh()
        )
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *ape)
        return img

def trainToplevel():
    generator = Generator()
    generator.load_state_dict(torch.load('./generator-32000.pt', map_location="cpu")) #'./images_post_top/model-1330.pt', map_location="cpu"))

    ''' Diagramlayer Toplevel Setup'''
    dtype=torch.float32
    width, height = 28, 28
    axis_x = np.arange(0, width)
    axis_y = np.arange(0, height)
    grid_axes = np.array(np.meshgrid(axis_x, axis_y))
    grid_axes = np.transpose(grid_axes, (1, 2, 0))
    from scipy.spatial import Delaunay
    tri = Delaunay(grid_axes.reshape([-1, 2]))
    faces = tri.simplices.copy()
    F = DiagramlayerToplevel().init_filtration(faces)
    diagramlayerToplevel = DiagramlayerToplevel.apply
    ''' '''

    z = torch.autograd.Variable(torch.Tensor(np.random.normal(0, 1, (28, 100))), requires_grad=True)
    with torch.no_grad():
        gen_image = generator(z)
        save_image(gen_image.data[:25], 'toplevel_before.png', nrow=5, normalize=False)

    lr = 0.01
    #optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer = torch.optim.Adam([z], lr=lr)
    for i in range(10):
        optimizer.zero_grad()
        #z = torch.Tensor(np.random.normal(0, 1, (28, 100)))
        gen_image = generator(z)
        top_loss = top_batch_cost(gen_image, diagramlayerToplevel, F)
        top_loss.backward()
        optimizer.step()
        print ("[Iter %d] [G loss: %f]" % (i, top_loss.item()))

    with torch.no_grad():
        #z = torch.Tensor(np.random.normal(0, 1, (28, 100)))
        gen_image = generator(z)
        save_image(gen_image.data[:25], 'toplevel_after.png', nrow=5, normalize=False)

def trainRips():
    ''' Rips setup '''
    diagramlayerRips = DiagramlayerRips.apply
    ''' '''

    ''' #### CIRCLE #### '''
    num_samples = 30
    # make a simple unit circle
    theta = np.linspace(0, 2*np.pi, num_samples)
    a, b = 1 * np.cos(theta), 1 * np.sin(theta)
    # generate the points
    theta = np.random.rand((num_samples)) * (2 * np.pi)
    r = 1.0 # np.random.rand((num_samples))
    x, y = r * np.cos(theta), r * np.sin(theta)
    circle = np.array([x,y]).reshape([len(x), 2])
    circle = (circle.T * (1.0 / np.linalg.norm(circle, axis=1))).T
    data = circle
    ''' #### END #### '''

    plt.figure()
    plt.scatter(data[:,0], data[:,1])
    plt.savefig('rips_before.png')

    saturation = 3.0 # Arbitrary
    var = torch.tensor(data, requires_grad=True, dtype=dtype)
    optimizer = torch.optim.Adam([var], lr = 0.01)
    for iter in range(10):
        optimizer.zero_grad()
        diagrams = diagramlayerRips(var, saturation)
        loss = cost_function(diagrams)
        loss.backward()
        optimizer.step()
        print ("[Iter %d] [loss: %f]" % (iter, loss.item()))

    data = var.detach().numpy()
    plt.figure()
    plt.scatter(data[:,0], data[:,1])
    plt.savefig('rips_after.png')

if __name__ == "__main__":
    trainToplevel()
    trainRips()
