from __future__ import print_function
import numpy as np
from ..util.flag_dionysus import computePersistence
import dionysus as d
import time
import torch
import itertools
from scipy.spatial import Delaunay
from torch.autograd import Variable, Function
dtype=torch.float32 # torch.double #torch.float32
PLOT = True


def alpha_filtration_1d(x):
    """
    returns dionysus filtration on 1D space
    """
    inds = np.argsort(x)
    simplices = []
    # append 0-cells
    for i in inds:
        simplices.append(([i], 0.))
    # append 1-cells
    for ii in range(len(inds) - 1):
        simplices.append(([inds[ii], inds[ii+1]], x[inds[ii+1]] - x[inds[[ii]]]))
    return d.Filtration(simplices)


def alpha_filtration(x, maxdim=2):
    """
    compute Delaunay triangulation
    fill in simplices as appropriate

    if x is 1-dimensional, defaults to 1D alpha
    inputs:
        x - pointcloud
        maxdim - maximal simplex dimension (default = 2)
    """
    if x.shape[1] == 1:
        x = x.flatten()
        return alpha_filtration_1d(x)
    tri = Delaunay(x)
    simplices = []
    # fill in 0 - cells
    for i in range(len(x)):
        simplices.append(([i], 0.))
    # fill in higher-d - cells
    for s in tri.simplices:
        # edges
        for i, j in itertools.combinations(s, 2):
            simplices.append(([i, j], np.linalg.norm(x[i] - x[j])))
        # higher order simplices
        for dim in range(2, maxdim+1):
            # loop over faces
            for face in itertools.combinations(s, dim+1):
                # get filtration value for face
                vface = 0.
                for i, j in itertools.combinations(face, 2):
                    vface = max(vface, np.linalg.norm(x[i] - x[j]))
                simplices.append((list(face), vface))
    return d.Filtration(simplices)


''' OBS: -1.0 are used as a token value for dgm values and indicies!!!!!! '''
class Diagramlayer(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, x, maxdim=1, verbose=False):
        SKELETON_DIMENSION = maxdim + 1 # maximal simplex dimension
        if verbose: print("*** dgm start")

        start_time = time.time()
        function_values = x
        # list of function values on vertices, and maximal dimension it will return 0,1,2,3
        function_useable = function_values.data.numpy()
        # F = d.fill_rips(function_useable, MAX_DIMENSION, SATURATION_VALUE)
        F = alpha_filtration(function_useable, maxdim=SKELETON_DIMENSION)
        # F.sort() # this is done in computePersistence

        dgms, Tbl = computePersistence(F)
        max_pts = np.max([len(dgms[i]) for i in range(maxdim+1)])
        num_dgm_pts = max_pts
        ''' -1 is used later '''
        dgms_inds = -1 * np.ones([maxdim+1, num_dgm_pts, 4])
        dgms_values = -np.inf * np.ones([maxdim+1, num_dgm_pts, 2]) # -1.0 * np.ones([3, num_dgm_pts, 2])
        for dim in range(maxdim+1):
            if len(dgms[dim]) > 0:
                dgm = np.array(dgms[dim])
                l = np.min([num_dgm_pts, len(dgm)])
                arg_sort = np.argsort(np.abs(dgm[:,1] - dgm[:,0]))[::-1]
                dgms_inds[dim][:l] = dgm[arg_sort[:l], 2:6]
                dgms_values[dim][:l] = dgm[arg_sort[:l], 0:2]

        dgms_inds = dgms_inds.reshape([maxdim+1, num_dgm_pts, 2, 2])
        #print dgms_values
        #dgms_values[dgms_values == np.inf] = SATURATION_VALUE #-1.0, Won't show up as inifinite, but good enough
        output = torch.tensor(dgms_values).type(dtype)
        ctx.save_for_backward(x, torch.tensor(dgms_inds).type(dtype), output, torch.tensor(verbose))
        if verbose: print("*** dgm done", time.time() - start_time)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, dgms_inds, dgms_values, verbose = ctx.saved_variables
        if verbose: print("*** dgm back")
        start_time = time.time()

        points = input.data.numpy()
        output = dgms_values.detach().numpy()
        grad_input = torch.zeros(input.shape).type(dtype)

        # MASK to only care about relevant spots later one
        output[output == np.inf] = -np.inf # death_value infinite doesn't correspond to a simplex
        output[output > -np.inf] = 1 # actual values that map to simplices
        output[output == -np.inf] = 0 # float('NaN') # 0 # dont affect the gradient, since they dont exist, didn't have matches, just because we want to keep matrix structure

        np_dgms_inds = dgms_inds.data.numpy().astype(np.int) # (3, 18424, 2, 2)
        # print np_dgms_inds.shape # (3, 18424, 4)
        list_of_unique_indices = np.unique(np_dgms_inds.flatten())
        grad_intermediate = output * grad_output.detach().numpy() # Not necessary? (dgms, dgm_pts, 2)
        ''' will have incorrect mappings, but these will never be used? '''
        pts_of_inds = points[np_dgms_inds]
        #print "pts_of_inds", pts_of_inds.shape # (3, 50, 2, 2, 2)

        for i in range(len(list_of_unique_indices)):
            index = int(list_of_unique_indices[i]) # index into input, get all that responds to a point-index
            ''' Not mapped anyhwere, set above '''
            if index > -1:
                index_into_dgms_inds = np.argwhere(np_dgms_inds == index)
                index_into_dgms_inds = index_into_dgms_inds.transpose()
                index_into_dgms_inds_partners = np.copy(index_into_dgms_inds)
                index_into_dgms_inds_partners[-1, :] = np.remainder(index_into_dgms_inds[-1, :] + 1, 2)
                intermediate = pts_of_inds[list(index_into_dgms_inds)] - pts_of_inds[list(index_into_dgms_inds_partners)] #- dgms_inds_to_points[np.remainder(np.array(index_into_dgms_inds)+1, 2)]
                ''' No 1.0/2 factor for dionysus '''
                #print("intermediate", intermediate)
                ''' Dividing by np.linalg.norm for zero norm has unintended consequences '''
                norms = np.linalg.norm(intermediate, axis=1)
                norms[norms == 0] = 1.0
                intermediate = ( intermediate.transpose() / norms).transpose()
                inds_into_grad_output = index_into_dgms_inds[:-1, :]
                grad_output_and_intermediate = (intermediate.transpose() * grad_intermediate[ list(inds_into_grad_output) ]).transpose()
                update = np.sum( grad_output_and_intermediate.reshape([-1, input.shape[1]]), axis=0 )
                grad_input[int(index)] = torch.tensor(update).type(dtype)
        if verbose: print("*** dgm back done", time.time() - start_time)
        return grad_input, None, None, None

if __name__ == "__main__":
    diagramlayer = Diagramlayer.apply
    from torch.autograd import gradcheck
    from utils_plot import plot_diagram2
    from scipy.spatial import Delaunay

    ''' #### Generate initial points #### '''
    import matplotlib.pyplot as plt
    np.random.seed(0)
    num_samples = 30 # 2048
    # make a simple unit circle
    theta = np.linspace(0, 2*np.pi, num_samples)
    a, b = 1 * np.cos(theta), 1 * np.sin(theta)
    # generate the points
    theta = np.random.rand((num_samples)) * (2 * np.pi)
    r = 1.0 # np.random.rand((num_samples))
    x, y = r * np.cos(theta), r * np.sin(theta)
    circle = np.array([x,y]).reshape([len(x), 2])
    circle = (circle.T * (1.0 / np.linalg.norm(circle, axis=1))).T
    #print circle
    plt.figure()
    plt.scatter(circle[:,0], circle[:,1])
    plt.savefig('CIRCLE.png')
    ''' #### END #### '''

    ''' #### Rips #### '''
    # f = d.fill_rips(circle, 2, 2.1)
    # f.sort()
    # gradchek takes a tuple of tensor as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.


    layer = Diagramlayer.apply
    ''' #### Test #### '''
    weights = Variable(torch.tensor(circle).type(dtype), requires_grad=True)

    # diagramlayer = Diagramlayer.apply
    # dgms = diagramlayer(weights)
    # dgms = dgms.detach().numpy()
    # print dgms
    # for d_i in range(dgms.shape[0]):
    #
    #     dgmpts = dgms[d_i]
    #     print dgmpts.shape
    #     dgmpts = np.delete(dgmpts, np.where((dgmpts == (-np.inf, -np.inf)).all(axis=1)), axis=0)
    #     dgmpts0 = dgmpts
    #     if len(dgmpts) > 0:
    #         fig = plot_diagram2(dgmpts, 'Dimension {}'.format(0))
    #     else:
    #         fig = plt.figure()
    #     fig.savefig('dgm{}_{}.png'.format(d_i, "test"))

    saturation = 1.1
    input = (weights, saturation)
    test = gradcheck(layer, input, eps=1e-4, atol=1e-3)
    print(test)
