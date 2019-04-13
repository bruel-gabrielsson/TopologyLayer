import tensorflow as tf
import numpy as np
import sys
import topologicalutilsRIPS as tp
import dionysus as d
import time
from t_utils_plot import plot_diagram2
from tf_rips import tf_rips, tf_rips_py

dtype=np.float32

def plot_tf(dgms, save_path="./rips", name="1"):
    for d_i in range(dgms.shape[0]):
        dgmpts = dgms[d_i]
        #print dgmpts.shape
        dgmpts = np.delete(dgmpts, np.where((dgmpts < (0, 0)).all(axis=1)), axis=0)
        dgmpts[dgmpts < 0] = np.inf
        dgmpts0 = dgmpts
        if len(dgmpts) > 0:
            fig = plot_diagram2(dgmpts, 'Dimension {}'.format(0))
        else:
            fig = plt.figure()
        fig.savefig(save_path + 'dgm{}_{}.png'.format(d_i, name))

def tf_cost(y):
    #print "SSSS", y[0,1,1].eval(), y[0,1,0].eval()
    return tf.square(y[0,1,1] - y[0,1,0])

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

    # a = tf_rips_py(tf_pts)
    # cost = tf_cost(a)

    # Add an Op to initialize global variables.
    init_op = tf.global_variables_initializer()
    # Launch the graph in a session.
    # opt = tf.train.GradientDescentOptimizer(0.01) #.minimize(cost)
    with tf.Session() as sess:
        # Run the Op that initializes global variables.
        sess.run(init_op)
        # ...you can now run any Op that uses variable values...

        a = tf_rips(tf_pts)
        cost = tf_cost(a)
        # dy = tf.gradients(y, x)
        # dy = tf.gradients(cost, a)

        #a = tf.reshape(a, [-1])

        opt = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

        for i in range(100):
            #print "dy", dy[0].eval()
            #a = tf_rips(tf_pts)
            #cost = tf_cost(a)
            #grads = opt.compute_gradients(cost)
            #grad_vals = sess.run(grads)
            #print "grad_vals", grad_vals, "end grad_vals"
            #apply_op = opt.apply_gradients(grads)
            #sess.run(apply_op)

            #grads = sess.run(grads) #[grad[0] for grad in grads])
            #print "grad_vals", grads, "end grad_vals"
            #sess.run(grads)
            #
            #sess.run(tf_pts)
            a = tf_rips(tf_pts)
            cost = tf_cost(a)
            #opt = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
            _, _cost = sess.run([cost, opt])
            print "_", a.eval(), _cost, _
            #print i, _, _cost
            print tf_pts.eval(), "COST", cost.eval()
            pts = tf_pts.eval()
            plt.figure()
            plt.scatter(pts[:,0], pts[:,1])
            plt.savefig('rips/points_{}.png'.format(i))
            plot_tf(a.eval(), './rips/', i)

        if False:
            print "output", a.eval()
            print "shapes", tf_pts.eval().shape, a.eval().shape
            # theoretical and numerical Jacobian
            check = tf.test.compute_gradient(tf_pts, tf_pts.eval().shape, a, a.eval().shape) #, x_init_value=circle)
            #print a.eval()
            err = tf.test.compute_gradient_error(tf_pts, tf_pts.eval().shape, a, a.eval().shape)
            print "check", check
            print  "error", err
