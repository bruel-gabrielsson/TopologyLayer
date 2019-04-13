import tensorflow as tf
import numpy as np
import sys
import topologicalutilsRIPS as tp
import dionysus as d
import time
from t_utils_plot import plot_diagram2

from tensorflow.python.framework import ops

# @tf.custom_gradient
# def log1pexp(x):
#   e = tf.exp(x)
#   def grad(dy):
#     return dy * (1 - 1 / (1 + e))
#   return tf.log(1 + e), grad

dtype=np.float32
PLOT = True

DGMS_INDS = []
DGMS_VALUES = []

def rips_dgms(x, saturation=2.0):
    #saturation = 2.0
    MAX_DIMENSION = 1
    if PLOT: print "*** dgm start"
    if saturation == None:
        SATURATION_VALUE = 3.1
        print "==== WARNING: NO SATURATION VALUE GIVEN, {}".format(SATURATION_VALUE)
    else:
        SATURATION_VALUE = saturation #.eval()

    #print SATURATION_VALUE
    start_time = time.time()
    function_values = x
    # list of function values on vertices, and maximal dimension it will return 0,1,2,3
    function_useable = np.array(function_values) #.eval() # session=tf.Session()) #eval() #data.numpy()
    print "function_useable", function_useable
    #print function_useable
    ''' 2 is max homology dimension '''
    F = d.fill_rips(function_useable, MAX_DIMENSION, SATURATION_VALUE)
    F.sort()
    dgms, Tbl = tp.computePersistence(F)
    max_pts = np.max([len(dgms[0]), len(dgms[1]), len(dgms[2])])
    num_dgm_pts = 3 # max_pts # max_pts # 1 #
    ''' -1 is used later '''
    dgms_inds = -1 * np.ones([3, num_dgm_pts, 4])
    dgms_values = -1.0 * np.ones([3, num_dgm_pts, 2]) # -np.inf * np.ones([3, num_dgm_pts, 2])
    if len(dgms[0]) > 0:
        dim = 0
        dgm = np.array(dgms[dim])
        dgm[dgm == np.inf] = SATURATION_VALUE
        l = np.min([num_dgm_pts, len(dgm)])
        arg_sort = np.argsort(np.abs(dgm[:,1] - dgm[:,0]))[::-1]
        dgms_inds[dim][:l] = dgm[arg_sort[:l], 2:6]
        dgms_values[dim][:l] = dgm[arg_sort[:l], 0:2]
    if len(dgms[1]) > 0:
        dim = 1
        dgm = np.array(dgms[dim])
        dgm[dgm == np.inf] = SATURATION_VALUE
        l = np.min([num_dgm_pts, len(dgm)])
        arg_sort = np.argsort(np.abs(dgm[:,1] - dgm[:,0]))[::-1] # TAKE MOST PERSISTENT
        dgms_inds[dim][:l] = dgm[arg_sort[:l], 2:6]
        dgms_values[dim][:l] = dgm[arg_sort[:l], 0:2]
        # l = np.min([num_dgm_pts, len(dgms[1])])
        # dgms_inds[1][:l] = np.array(dgms[1])[:l, [2,3,4,5]]
        # dgms_values[1][:l] = np.array(dgms[1])[:l, [0,1]]
    if False: # len(dgms[2]) > 0:
        dim = 2
        dgm = np.array(dgms[dim])
        dgm[dgm == np.inf] = SATURATION_VALUE
        l = np.min([num_dgm_pts, len(dgm)])
        arg_sort = np.argsort(np.abs(dgm[:,1] - dgm[:,0]))[::-1]
        dgms_inds[dim][:l] = dgm[arg_sort[:l], 2:6]
        dgms_values[dim][:l] = dgm[arg_sort[:l], 0:2]

    dgms_inds = dgms_inds.reshape([3, num_dgm_pts, 2, 2])
    if PLOT: print "*** dgm start done", time.time() - start_time
    return dgms_values, dgms_inds

def forward(x):
    global DGMS_INDS, DGMS_VALUES
    saturation = 2.0
    dgms_values, dgms_inds = rips_dgms(x, saturation)
    DGMS_INDS, DGMS_VALUES = dgms_inds, dgms_values
    return np.array(dgms_values).astype(dtype) #, np.array(dgms_inds)
    #return x + 1

def backward(op, grad_output):
    global DGMS_INDS, DGMS_VALUES
    x = op.inputs[0]
    #dgms_values = op.outputs[0]
    #dgms_inds = op.inputs[2]
    assert(len(DGMS_INDS) > 0)
    assert(len(DGMS_VALUES) > 0)
    dgms_inds, dgms_values = DGMS_INDS, DGMS_VALUES
    print "dgms_inds.shape, dgms_values.shape", dgms_inds.shape, dgms_values.shape
    # = op.outputs[1]
    #print op.outputs
    #output = op.outputs[0] # np.array()
    #print "output shape", output.shape
    #dgms_inds = output[:,:,2:6]
    #dgms_values = output[:,:,0:2]
    print "dgms_values", dgms_values.shape
    function_useable = dgms_values # x #np.array(x)
    if PLOT: print "*** dgm back"
    start_time = time.time()
    #input, dgms_inds, dgms_values = ctx.saved_variables
    #input = function_useable
    points = function_useable # input.data.numpy()
    output = dgms_values #.detach().numpy()
    grad_input = np.zeros(function_useable.shape) # tf.zeros(function_useable.shape, dtype=dtype) #torch.zeros(input.shape).type(dtype)

    # MASK to only care about relevant spots later one
    # output[output == np.inf] = -np.inf # death_value infinite doesn't correspond to a simplex
    # output[output > -np.inf] = 1 # actual values that map to simplices
    # output[output == -np.inf] = 0 # float('NaN') # 0 # dont affect the gradient, since they dont exist, didn't have matches, just because we want to keep matrix structure

    np_dgms_inds = np.array(dgms_inds) #.data.numpy().astype(np.int) # (3, 18424, 2, 2)
    # print np_dgms_inds.shape # (3, 18424, 4)
    list_of_unique_indices = np.unique(np_dgms_inds.flatten())
    #grad_intermediate = output * np.array(grad_output) #.eval() #detach().numpy() # Not necessary? (dgms, dgm_pts, 2)
    grad_intermediate = np.array(grad_output)
    ''' will have incorrect mappings, but these will never be used? '''
    pts_of_inds = points[np.array(np_dgms_inds).astype(np.int)]
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
            intermediate = ( intermediate.transpose() / np.linalg.norm(intermediate, axis=1) ).transpose()
            inds_into_grad_output = index_into_dgms_inds[:-1, :]
            grad_output_and_intermediate = (intermediate.transpose() * grad_intermediate[ list(inds_into_grad_output) ]).transpose()
            update = np.sum( grad_output_and_intermediate.reshape([-1, 2]), axis=0 )
            # if np.linalg.norm(update) == 0:
            #     print "update", update
            #     print "index, index_into_dgms_inds_partners", index, index_into_dgms_inds_partners
            #     print "index_into_dgms_inds", index_into_dgms_inds
            #     print "intermediate", intermediate
            #     print "inds_into_grad_output", inds_into_grad_output
            #     print "grad_intermediate[ list(inds_into_grad_output) ]", grad_intermediate[ list(inds_into_grad_output) ]
            #     print "grad_output, grad_intermediate", grad_output, grad_intermediate
            #     assert(False)
            grad_input[int(index)] = update # tf.convert_to_tensor(update, dtype) #torch.tensor(update).type(dtype)
    if PLOT: print "*** dgm back done", time.time() - start_time
    #return grad_input, None
    #grad_input = tf.convert_to_tensor(grad_input, dtype)
    #print grad_input.eval()
    return np.array(grad_input).astype(dtype)
    #return x

# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def tf_rips_py(x):
    return py_func(forward, [x], [dtype], grad=backward)[0]

@tf.custom_gradient
def tf_rips(x): #, saturation=None):
    saturation = 2.0
    MAX_DIMENSION = 1
    if PLOT: print "*** dgm start"
    if saturation == None:
        SATURATION_VALUE = 3.1
        print "==== WARNING: NO SATURATION VALUE GIVEN, {}".format(SATURATION_VALUE)
    else:
        SATURATION_VALUE = saturation #.eval()

    #print SATURATION_VALUE
    start_time = time.time()
    function_values = x
    # list of function values on vertices, and maximal dimension it will return 0,1,2,3
    function_useable = function_values.eval() # session=tf.Session()) #eval() #data.numpy()
    print "function_useable", function_useable
    #print function_useable
    ''' 2 is max homology dimension '''
    F = d.fill_rips(function_useable, MAX_DIMENSION, SATURATION_VALUE)
    F.sort()

    dgms, Tbl = tp.computePersistence(F)
    max_pts = np.max([len(dgms[0]), len(dgms[1]), len(dgms[2])])
    num_dgm_pts = 3 # max_pts # max_pts # 1 #
    ''' -1 is used later '''
    dgms_inds = -1 * np.ones([3, num_dgm_pts, 4])
    dgms_values = -1.0 * np.ones([3, num_dgm_pts, 2]) # -np.inf * np.ones([3, num_dgm_pts, 2])
    if len(dgms[0]) > 0:
        dim = 0
        dgm = np.array(dgms[dim])
        dgm[dgm == np.inf] = SATURATION_VALUE
        l = np.min([num_dgm_pts, len(dgm)])
        arg_sort = np.argsort(np.abs(dgm[:,1] - dgm[:,0]))[::-1]
        dgms_inds[dim][:l] = dgm[arg_sort[:l], 2:6]
        dgms_values[dim][:l] = dgm[arg_sort[:l], 0:2]
    if len(dgms[1]) > 0:
        dim = 1
        dgm = np.array(dgms[dim])
        dgm[dgm == np.inf] = SATURATION_VALUE
        l = np.min([num_dgm_pts, len(dgm)])
        arg_sort = np.argsort(np.abs(dgm[:,1] - dgm[:,0]))[::-1] # TAKE MOST PERSISTENT
        dgms_inds[dim][:l] = dgm[arg_sort[:l], 2:6]
        dgms_values[dim][:l] = dgm[arg_sort[:l], 0:2]
        # l = np.min([num_dgm_pts, len(dgms[1])])
        # dgms_inds[1][:l] = np.array(dgms[1])[:l, [2,3,4,5]]
        # dgms_values[1][:l] = np.array(dgms[1])[:l, [0,1]]
    if False: # len(dgms[2]) > 0:
        dim = 2
        dgm = np.array(dgms[dim])
        dgm[dgm == np.inf] = SATURATION_VALUE
        l = np.min([num_dgm_pts, len(dgm)])
        arg_sort = np.argsort(np.abs(dgm[:,1] - dgm[:,0]))[::-1]
        dgms_inds[dim][:l] = dgm[arg_sort[:l], 2:6]
        dgms_values[dim][:l] = dgm[arg_sort[:l], 0:2]

    dgms_inds = dgms_inds.reshape([3, num_dgm_pts, 2, 2])
    #print dgms_values
    #dgms_values[dgms_values == np.inf] = SATURATION_VALUE #-1.0, Won't show up as inifinite, but good enough
    output = tf.convert_to_tensor(dgms_values, np.float32) # torch.tensor(dgms_values).type(dtype)
    #ctx.save_for_backward(x, torch.tensor(dgms_inds).type(dtype), output)
    if PLOT: print "*** dgm done", time.time() - start_time
    #return output

    ''' GRADIENT '''
    def grad(grad_output):
        if PLOT: print "*** dgm back"
        start_time = time.time()
        #input, dgms_inds, dgms_values = ctx.saved_variables
        #input = function_useable
        points = function_useable # input.data.numpy()
        output = dgms_values #.detach().numpy()
        grad_input = np.zeros(function_useable.shape) # tf.zeros(function_useable.shape, dtype=dtype) #torch.zeros(input.shape).type(dtype)

        # MASK to only care about relevant spots later one
        output[output == np.inf] = -np.inf # death_value infinite doesn't correspond to a simplex
        output[output > -np.inf] = 1 # actual values that map to simplices
        output[output == -np.inf] = 0 # float('NaN') # 0 # dont affect the gradient, since they dont exist, didn't have matches, just because we want to keep matrix structure

        np_dgms_inds = dgms_inds #.data.numpy().astype(np.int) # (3, 18424, 2, 2)
        # print np_dgms_inds.shape # (3, 18424, 4)
        list_of_unique_indices = np.unique(np_dgms_inds.flatten())
        grad_intermediate = output * grad_output.eval() #detach().numpy() # Not necessary? (dgms, dgm_pts, 2)
        ''' will have incorrect mappings, but these will never be used? '''
        pts_of_inds = points[np_dgms_inds.astype(np.int)]
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
                intermediate = ( intermediate.transpose() / np.linalg.norm(intermediate, axis=1) ).transpose()
                inds_into_grad_output = index_into_dgms_inds[:-1, :]
                grad_output_and_intermediate = (intermediate.transpose() * grad_intermediate[ list(inds_into_grad_output) ]).transpose()
                update = np.sum( grad_output_and_intermediate.reshape([-1, 2]), axis=0 )
                # if np.linalg.norm(update) == 0:
                #     print "update", update
                #     print "index, index_into_dgms_inds_partners", index, index_into_dgms_inds_partners
                #     print "index_into_dgms_inds", index_into_dgms_inds
                #     print "intermediate", intermediate
                #     print "inds_into_grad_output", inds_into_grad_output
                #     print "grad_intermediate[ list(inds_into_grad_output) ]", grad_intermediate[ list(inds_into_grad_output) ]
                #     print "grad_output, grad_intermediate", grad_output, grad_intermediate
                #     assert(False)
                grad_input[int(index)] = update # tf.convert_to_tensor(update, dtype) #torch.tensor(update).type(dtype)
        if PLOT: print "*** dgm back done", time.time() - start_time
        #return grad_input, None
        grad_input = tf.convert_to_tensor(grad_input, dtype)
        print grad_input.eval()
        return grad_input

    return output, grad

if __name__ == "__main__":
    ''' #### Generate initial points #### '''
    import matplotlib.pyplot as plt
    np.random.seed(0)
    num_samples = 2 # 2048
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
    saturation = 1.0

    tf_pts = tf.Variable( tf.convert_to_tensor(circle, dtype) )
    #a = tf_rips_py(tf_pts)
    # Add an Op to initialize global variables.
    init_op = tf.global_variables_initializer()
    # Launch the graph in a session.

    ''' py_func '''
    # with tf.Session() as sess:
    #     # Run the Op that initializes global variables.
    #     sess.run(init_op)
    #     # ...you can now run any Op that uses variable values...
    #
    #     #a = tf.reshape(a, [-1])
    #     print "output", a.eval()
    #     print "shapes", tf_pts.eval().shape, a.eval().shape
    #
    #     # theoretical and numerical Jacobian
    #     check = tf.test.compute_gradient(tf_pts, tf_pts.eval().shape, a, a.eval().shape) #, x_init_value=circle)
    #     #print a.eval()
    #     err = tf.test.compute_gradient_error(tf_pts, tf_pts.eval().shape, a, a.eval().shape)
    #     print "check", check
    #     print  "error", err

    ''' PREV '''
    with tf.Session() as sess:
        # Run the Op that initializes global variables.
        sess.run(init_op)
        # ...you can now run any Op that uses variable values...
        a = tf_rips(tf_pts)
        #a = tf.reshape(a, [-1])
        print "output", a.eval()
        print "shapes", tf_pts.eval().shape, a.eval().shape
        #_, _cost = sess.run([opt, cost])
        # theoretical and numerical Jacobian
        check = tf.test.compute_gradient(tf_pts, tf_pts.eval().shape, a, a.eval().shape) #, x_init_value=circle)
        #print a.eval()
        err = tf.test.compute_gradient_error(tf_pts, tf_pts.eval().shape, a, a.eval().shape)
        print "check", check
        print  "error", err
