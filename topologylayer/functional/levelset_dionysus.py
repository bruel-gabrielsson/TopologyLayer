from __future__ import print_function
import numpy as np
# import sys
# sys.path.append('../Python')
from ..util.star_dionysus import computePersistence
import dionysus as d
import time
import torch
from torch.autograd import Variable, Function
dtype=torch.float32
''' Change to touch.double to try gradient check '''
# dtype=torch.double
PRINT = False

class Diagramlayer(Function):
    def __init__(self):
        super(Diagramlayer, self).__init__()
        # self.p = chopt.Cohomology()
        # self.p.readIn(triangulation_file)
        # self.p.init()

    def init_filtration(self, faces):
        F = faces
        f = d.Filtration()
        for i in range(F.shape[0]):
            if len(F[i]) == 4:
                c = d.closure([d.Simplex([int(F[i][0]), int(F[i][1]), int(F[i][2]), int(F[i][3])], 0.0)], 3) #np.array(0).astype(DTYPE))],2)
            elif len(F[i]) == 3:
                c = d.closure([d.Simplex([int(F[i][0]), int(F[i][1]), int(F[i][2])], 0.0)], 3)
            for j in c:
                f.append(j)
        return f

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, x, F):
        if PRINT: print("*** dgm start")
        start_time = time.time()
        function_values = x
        # list of function values on vertices, and maximal dimension it will return 0,1,2,3
        function_useable = function_values.data.numpy().flatten()

        dgms, Tbl = computePersistence(F, -1.0 * function_useable.reshape(1,-1))
        num_dgm_pts = np.max([len(dgms[0]), len(dgms[1]), len(dgms[2])])
        dgms_inds = -1 * np.ones([3, num_dgm_pts, 2])
        if len(dgms[0]) > 0:
            dgms_inds[0][:len(dgms[0])] = np.array(dgms[0])[:, [2,3]]
        if len(dgms[1]) > 0:
            dgms_inds[1][:len(dgms[1])] = np.array(dgms[1])[:, [2,3]]
        if len(dgms[2]) > 0:
            dgms_inds[2][:len(dgms[2])] = np.array(dgms[2])[:, [2,3]]
        dgms_inds[np.where(dgms_inds == np.inf)] = -1

        vertices_indices = dgms_inds
        corr_function_values = -np.inf * np.ones(vertices_indices.shape)
        dgm = vertices_indices
        for i in range(0, len(dgm)):
            if len(dgm[i]) > 0:
                dgmi = np.unique( np.array(dgm[i]), axis=0 ).astype(np.int)
                corr_f_values = function_useable[dgmi]
                ''' Change to -1.0 for gradient check '''
                corr_f_values[np.where(dgmi == -1)] = -np.inf # -1.0
                corr_function_values[i][:np.min([corr_f_values.shape[0], num_dgm_pts])] = corr_f_values[:np.min([corr_f_values.shape[0], num_dgm_pts])]
                vertices_indices[i][:np.min([len(dgmi), num_dgm_pts])] = dgmi[:np.min([len(dgmi), num_dgm_pts])]

        output = torch.tensor(corr_function_values).type(dtype)
        ctx.save_for_backward(x, torch.tensor(vertices_indices).type(dtype), output)
        if PRINT: print("*** dgm done", time.time() - start_time)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, vertices_indices, output = ctx.saved_variables
        grad_input = torch.zeros(input.shape).type(dtype)
        output[output > -np.inf] = 1
        output[output == -np.inf] = 0
        np_vertices_indices = vertices_indices.numpy()
        list_of_unique_indices = np.unique(np_vertices_indices.flatten())
        grad_intermediate = output * grad_output # (dgms, dgm_pts, 2)
        for i in range(len(list_of_unique_indices)):
            index = list_of_unique_indices[i]
            if index > -1:
                index_into_grad_output = np.where(np_vertices_indices == index)
                grad_input[int(index)] = torch.sum( grad_intermediate[index_into_grad_output] )
        return grad_input, None

if __name__ == "__main__":
    ''' Change above to touch.double and to -1.0 (a constant) to try gradient check '''
    ''' This is because double is needed for precision and a constant for well defined numerical gradients '''
    ''' But in practice we use float32 for speed and -np.inf for ease of use '''
    diagramlayer = Diagramlayer.apply
    from torch.autograd import gradcheck
    layer = Diagramlayer.apply # ("./test1.off")
    from scipy.spatial import Delaunay
    points = np.array([[0.01, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [0, -0.1, 0.9]])
    tri = Delaunay(points)
    faces = tri.simplices.copy()
    F = Diagramlayer().init_filtration(faces)
    # gradchek takes a tuple of tensor as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    input = (Variable(torch.randn(5,1).double(), requires_grad=True), F)
    test = gradcheck(layer, input, eps=1e-6, atol=1e-4)
    print(test)
