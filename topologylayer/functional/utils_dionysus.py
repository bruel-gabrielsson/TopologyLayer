from __future__ import print_function
import torch
import numpy as np
import time
dtype = torch.float32

''' Example of how to construct a cost function based on the persistence diagrams. '''
def cost_function(dgms):
    ''' Undefined are -1.0 for our rips and infinite are at saturation value '''
    dgm0 = dgms[0].type(dtype).clone()
    min_value = -1.0 # not always true, but can be assumed in most contexts
    dgm0[dgm0 == -np.inf] = min_value
    NAN = torch.tensor(float('nan')).type(dtype)
    lifetimes0 = torch.abs(dgm0[:,1]-dgm0[:,0])
    lifetimes0[lifetimes0 != lifetimes0] = 0
    sorted_d_dgm0, indsD0 = torch.sort( dgm0[:,1][dgm0[:,1] > -np.inf], 0)
    sorted0, inds0 = torch.sort(lifetimes0, 0, descending=True)

    cost = torch.add(
        torch.Tensor([0.0]),
        torch.mul(torch.sum(torch.abs(sorted0[1:1+10000])), 1.0),
        #torch.mul(torch.sum(torch.abs(sorted1[0:0+10000])), 1.0),
    )
    return cost

def top_cost(W, diagramlayer, filtration):
    dgms = diagramlayer(W.view(-1), filtration)
    return cost_function(dgms)

''' Example of how to run the cost over a batch '''
def top_batch_cost(gen_imgs, diagramlayer, filtration):
    start_time = time.time()
    axis=0
    costs = torch.stack([
        top_cost(x_i.view(-1), diagramlayer, filtration) for i, x_i in enumerate(torch.unbind(gen_imgs, dim=axis), 0)
    ], dim=axis)
    avg = torch.mean(costs.view(-1))
    print("top_batch_cost", "time: ", time.time() - start_time, "cost: ", avg)
    return avg
    ''' *** End Topology *** '''

def maxPolys(dgm1):
    b, d = dgm1[:, np.arange(0, dgm1.shape[1]-1,2)], dgm1[:, np.arange(1, dgm1.shape[1],2)]
    lifetimes = torch.abs(b - d)
    sorted, argsmax = torch.sort(lifetimes, dim=1, descending=True)
    #print sorted.shape, argsmax.shape # (64, 27)
    #argsmax = np.argsort(lifetimes)[::-1]
    #p1,p2,p3,p4 = np.sum(lifetimes[argsmax[:1]]),np.sum(lifetimes[argsmax[:2]]),np.sum(lifetimes[argsmax[:3]]),np.sum(lifetimes[argsmax[:4]])
    #print torch.sum(sorted[:,:2], dim=1).shape
    polys = torch.stack([
            torch.unsqueeze(torch.sum(sorted[:,:1], dim=1), dim=1),
            torch.unsqueeze(torch.sum(sorted[:,:2], dim=1), dim=1),
            torch.unsqueeze(torch.sum(sorted[:,:3], dim=1), dim=1),
            torch.unsqueeze(torch.sum(sorted[:,:4], dim=1), dim=1),
        ], dim=1)

    #print polys.shape # (64,4,1)
    polys = polys[:,:,0]
    return polys

def sumPolys(dgm1):
    b, d = dgm1[:, np.arange(0, dgm1.shape[1]-1,2)], dgm1[:, np.arange(1, dgm1.shape[1],2)]
    lifetimes = torch.abs(b - d)
    sorted, argsmax = torch.sort(lifetimes, dim=1, descending=True)
    polys = torch.unsqueeze(torch.sum(sorted, dim=1), dim=1)
    return polys

def top_features(x, diagramlayer, filtration, dim=1):
    # dgmsTop = diagramlayer(x.view(-1), filtration)
    # dgmsTop = dgmsTop.detach().numpy()
    # dgm1 = dgmsTop[1]
    # dgm1 = np.delete(dgm1, np.where((dgm1 == (-np.inf, -np.inf)).all(axis=1)), axis=0)
    # dgm1 = dgm1.flatten()
    #print("top_features", x)
    dgms = diagramlayer(x.view(-1), filtration)
    dgm1 = dgms[dim].type(dtype).clone()
    lifetimes1 = torch.abs(dgm1[:,1]-dgm1[:,0])
    lifetimes1[lifetimes1 != lifetimes1] = 0
    #sorted_d_dgm1, indsD1 = torch.sort( dgm1[:,1][dgm1[:,1] > -np.inf], 0)
    sorted1, inds1 = torch.sort(lifetimes1, 0, descending=True)
    p1, p2, p3, p4 = torch.sum(sorted1[:1]), torch.sum(sorted1[:2]), torch.sum(sorted1[:3]), torch.sum(sorted1[:4])
    #p1, p2, p3, p4 = torch.sum(sorted1[1:2]), torch.sum(sorted1[1:3]), torch.sum(sorted1[1:4]), torch.sum(sorted1[1:10])
    #print p1, p2, p3, p4
    polys = torch.stack([
            torch.unsqueeze(p1, dim=0),
            torch.unsqueeze(p2, dim=0),
            torch.unsqueeze(p3, dim=0),
            torch.unsqueeze(p4, dim=0),
        ], dim=1)
    #print polys
    polys = polys[0]
    return polys

def top_features_01(x, diagramlayer, filtration):
    dgms = diagramlayer(x.view(-1), filtration)
    ''' dim 0 '''
    dgm = dgms[0].type(dtype).clone()
    lifetimes = torch.abs(dgm[:,1]-dgm[:,0])
    lifetimes[lifetimes != lifetimes] = 0
    sorted, inds = torch.sort(lifetimes, 0, descending=True)
    a1, a2, a3, a4 = torch.sum(sorted[1:2]), torch.sum(sorted[1:3]), torch.sum(sorted[1:4]), torch.sum(sorted[1:10])
    ''' dim 1 '''
    dgm = dgms[1].type(dtype).clone()
    lifetimes = torch.abs(dgm[:,1]-dgm[:,0])
    lifetimes[lifetimes != lifetimes] = 0
    sorted, inds = torch.sort(lifetimes, 0, descending=True)
    p1, p2, p3, p4 = torch.sum(sorted[:1]), torch.sum(sorted[:2]), torch.sum(sorted[:3]), torch.sum(sorted[:4])
    polys = torch.stack([
            torch.unsqueeze(a1, dim=0),
            torch.unsqueeze(a2, dim=0),
            torch.unsqueeze(a3, dim=0),
            torch.unsqueeze(a4, dim=0),
            torch.unsqueeze(p1, dim=0),
            torch.unsqueeze(p2, dim=0),
            torch.unsqueeze(p3, dim=0),
            torch.unsqueeze(p4, dim=0),
        ], dim=1)
    #print polys
    polys = polys[0]
    return polys

def top_batch_features(input, diagramlayer, filtration, dim=1):
    #print(gen_imgs.shape)
    start_time = time.time()
    axis=0
    #print("input",input)
    feats = torch.stack([
        top_features(x_i.view(-1), diagramlayer, filtration, dim) for i, x_i in enumerate(torch.unbind(input, dim=axis), 0)
    ], dim=axis)
    #avg = torch.mean(costs.view(-1))
    print("feats", "time: ", time.time() - start_time)
    #print feats.shape
    return feats
