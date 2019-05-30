from __future__ import print_function
import numpy as np
#import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
from mpl_toolkits.mplot3d import Axes3D

def plot_dgms_top(dgms):
    figures = ()
    for i in range(len(dgms)):
        dgmpts = dgms[i]
        dgmpts = np.delete(dgmpts, np.where((dgmpts == (-np.inf, -np.inf)).all(axis=1)), axis=0)
        if len(dgmpts) > 0:
            fig = plot_diagram2(dgmpts, 'Dimension {}'.format(i))
        else:
            fig = plt.figure()
        figures = figures + (fig,)
    return figures

def plot_dgms_rips(dgms):
    figures = ()
    for i in range(len(dgms)):
        dgmpts = dgms[i]
        dgmpts = np.delete(dgmpts, np.where((dgmpts < (0, 0)).all(axis=1)), axis=0)
        dgmpts[dgmpts < 0] = np.inf
        if len(dgmpts) > 0:
            fig = plot_diagram2(dgmpts, 'Dimension {}'.format(i))
        else:
            fig = plt.figure()
        figures = figures + (fig,)
    return figures

def get_plot_3d(pts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') # plt.subplot(1, 2, 2)
    ax.scatter(pts[:,0].flatten(), pts[:,1].flatten(), pts[:,2].flatten(), c='b')
    return fig

def save_simp_complex(name, points, filt_function, func_values, diagramlayer, P, path='../../results_reconstruct/'):
    dgms = diagramlayer(func_values, P)
    dgms = dgms.numpy()
    func = func_values.numpy()
    # [[0.0, 0.0], [1.0, 3.0], [0.0, 1.0, 3.0], [2.0, 5.0], [0.0, 2.0, 5.0], [1.0, 2.0, 5.0], [0.0, 1.0, 2.0, 5.0]]
    filt_value = filt_function(dgms)
    np.savetxt(path + 'filtration_value_{}.txt'.format(name), np.array([filt_value]), delimiter=",")
    complex = P.printComplexOrder(-1.0 * func)
    tetrahedra = []
    vertices = []
    for i in range(len(complex)):
        c = complex[i]
        dim = len(c) - 2
        f_c = -1.0 * c[-1]
        if f_c >= filt_value:
            if dim > 0:
                tetrahedra.append( map(int, c[:(dim+1)]) )

    save_to_obj(points, tetrahedra, path + 'complex_{}.obj'.format(name))
    print("#### Saved complex")
    return filt_value

def save_simp_complex_dio(name, points, filt_function, func_values, diagramlayer, P, path='../../results_reconstruct/'):
    dgms = diagramlayer(func_values, P)
    dgms = dgms.numpy()
    func = func_values.numpy()
    # [[0.0, 0.0], [1.0, 3.0], [0.0, 1.0, 3.0], [2.0, 5.0], [0.0, 2.0, 5.0], [1.0, 2.0, 5.0], [0.0, 1.0, 2.0, 5.0]]
    filt_value = filt_function(dgms)
    np.savetxt(path + 'filtration_value_{}.txt'.format(name), np.array([filt_value]), delimiter=",")
    tetrahedra = []
    vertices = []
    print("#### dio filt_value", filt_value)

    dimension = points.shape[1]

    f = P
    for s in f:
        if -1.0 * s.data >= filt_value:
            # if s.dimension() == 3:
            #     tetrahedra.append([s[0], s[1], s[2], s[3]])
            if s.dimension() == 2 and dimension == 3:
                tetrahedra.append([s[0], s[1], s[2]])
            if s.dimension() == 1 and dimension == 2:
                tetrahedra.append([s[0], s[1]])
            # if s.dimension() == 1:
            #     tetrahedra.append([s[0], s[1]])

    save_to_obj(points, tetrahedra, path + 'complex_{}.obj'.format(name))

    print("#### Saved complex")
    return tetrahedra

''' Good for plotting persistent homology diagrams '''
def plot_diagram2(plot_data, title='Dimension 0'):
    fig = plt.figure()
    if len(plot_data) == 0: return fig
    infs = plot_data[plot_data[:,1] == np.inf]
    neg_infs = plot_data[plot_data[:,1] == -np.inf]
    infs = np.concatenate( (infs, neg_infs), axis=0 )
    num_infs = infs.shape[0]
    plot_data = plot_data[plot_data[:,1] < np.inf] # remove infs
    plot_data = plot_data[plot_data[:,1] > -np.inf]

    if len(plot_data) > 0: # in case their is only one inf and nothing else
        order_data = np.abs(plot_data[:,1] - plot_data[:,0]) # abs(death - birth)
        args = np.argsort(order_data)[::-1]
        plot_data = plot_data[args]
        plot_data = plot_data[:500] # only show top based on abs(death - birth) time
        max_value = np.max([np.max(plot_data[:,1]), np.max(plot_data[:,0])])
        min_value = np.min([np.min(plot_data[:,0]), np.min(plot_data[:,1])])
        if num_infs > 0:
            min_value = np.min([min_value, np.min(infs[:,0])]) # include so that we can see inf
            max_value = np.max([max_value, np.max(infs[:,0])])
        eps = np.abs(max_value - min_value) / 20.0
        lims = [min_value - eps, max_value + eps, min_value - eps, max_value + eps]
        plt.axis(lims)
        plt.plot([min_value, max_value], [min_value, max_value], color='k', linestyle='-', linewidth=2)
        plt.plot(plot_data[:,0], plot_data[:,1], "o")

    for j in range(0, infs.shape[0]):
        pt = infs[j,:]
        plt.plot([pt[0]], [pt[0]], "o", color="r")

    plt.title('{}. Num pts: {}. Num infs: {}'.format(title, len(plot_data), num_infs))
    plt.xlabel('birth')
    plt.ylabel('death')
    #plt.show()
    return fig
    #fig.savefig('{}/diagram_dim{}.png'.format(sub_dir, i))

# SHOULD BE INDEXED FROM 1 forward
def save_to_obj(coords, faces, name):
    f = open(name, "w")
    #faces = faces
    dim = coords.shape[1]
    for r in range(coords.shape[0]):
        row = coords[r]
        if dim == 2:
            f.write("v {} {} {}\n".format(row[0], row[1], 0))
        else:
            f.write("v {} {} {}\n".format(row[0], row[1], row[2]))
    if dim == 2:
        for fi in range(len(faces)):
            face = faces[fi]
            if len(face) == 2:
                f.write("l {} {}\n".format(face[0] + 1, face[1] + 1))
    elif dim == 3:
        for fi in range(len(faces)):
            face = faces[fi]
            # if len(face) == 2:
            #     f.write("l {} {}\n".format(face[0] + 1, face[1] + 1))
            if len(face) == 3:
                f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))
            # elif len(face) == 4:
            #     f.write("f {} {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1, face[3] + 1))

    f.close()
