import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
from t_utils import save_to_obj

#path = '../../results_reconstruct/'

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
    print "#### filt_value", filt_value, len(complex)

    for i in range(len(complex)):
        c = complex[i]
        dim = len(c) - 2
        f_c = -1.0 * c[-1]
        if f_c >= filt_value:
            if dim > 0:
                tetrahedra.append( map(int, c[:(dim+1)]) )
            # if dim == 0:
            #     vertices.append([])

    save_to_obj(points, tetrahedra, path + 'complex_{}.obj'.format(name))

    print "#### Saved complex"
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
    print "#### dio filt_value", filt_value

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

    print "#### Saved complex"
    return tetrahedra

def plot_and_save_info_3D(functionforward, diagramlayer, P, W, faces, centers, points, dimension, entropy, probability, t_kthdistance, name="", factor=1.0, path='../../results_reconstruct/'):
    # with torch.no_grad():
    print "#### Plot Save 3D"
    start_plot = time.time()
    #save_to_obj(points, faces, path + 'grid_{}.obj'.format(name))
    func = functionforward(W, centers, points, dimension, entropy, probability, t_kthdistance)

    np.savetxt(path + 'grid_{}.txt'.format(name), points, delimiter=",")

    dgms = diagramlayer(func, P)
    dgms = dgms.numpy()
    np.savetxt(path + 'weights_{}.txt'.format(name), W, delimiter=',')

    func = func.numpy()

    np.savetxt(path + 'function_{}.txt'.format(name), func, delimiter=',')

    dgmpts = dgms[0]
    dgmpts = np.delete(dgmpts, np.where((dgmpts == (-np.inf, -np.inf)).all(axis=1)), axis=0)
    dgmpts0 = dgmpts
    if len(dgmpts) > 0:
        fig = plot_diagram2(dgmpts, 'Dimension {}'.format(0))
    else:
        fig = plt.figure()
    fig.savefig(path + 'dgm0_{}.png'.format(name))

    dgmpts = dgms[1]
    dgmpts = np.delete(dgmpts, np.where((dgmpts == (-np.inf, -np.inf)).all(axis=1)), axis=0)
    dgmpts1 = dgmpts
    if len(dgmpts) > 0:
        fig = plot_diagram2(dgmpts, 'Dimension {}'.format(1))
        fig.savefig(path + 'dgm1_{}.png'.format(name))
    else:
        fig = plt.figure()
        fig.savefig(path + 'dgm1_{}.png'.format(name))

    dgmpts = dgms[2]
    dgmpts = np.delete(dgmpts, np.where((dgmpts == (-np.inf, -np.inf)).all(axis=1)), axis=0)
    dgmpts2 = dgmpts
    if len(dgmpts) > 0:
        fig = plot_diagram2(dgmpts, 'Dimension {}'.format(2))
        fig.savefig(path + 'dgm2_{}.png'.format(name))
    else:
        fig = plt.figure()
        fig.savefig(path + 'dgm2_{}.png'.format(name))

    print "#### Calc and plot persistence took {} seconds".format(time.time() - start_plot)

    # NUM_HOM_CARE_ABOUT = 1 # 3
    # # pts1 = np.concatenate((dgmpts1[:dgmpts1.shape[0]//2, [0]], dgmpts1[dgmpts1.shape[0]//2:, [0]]), axis=1)
    # # pts1 = -pts1
    # # sorted_hom_relevant = np.argsort( np.abs(pts1[:,1] - pts1[:,0] ) )[::-1][:NUM_HOM_CARE_ABOUT]
    # # intersection_homs = [np.max(pts1[sorted_hom_relevant, 0]), np.min(pts1[sorted_hom_relevant, 1])]
    # pts2 = dgmpts2
    # pts2 = -pts2
    # sorted_hom_relevant = np.argsort( np.abs(pts2[:,1] - pts2[:,0] ) )[::-1][:NUM_HOM_CARE_ABOUT]
    # intersection_homs = [np.max(pts2[sorted_hom_relevant, 0]), np.min(pts2[sorted_hom_relevant, 1])]
    # pts0 = dgmpts0
    # pts0 = -pts0 # So it's consistent with homclass
    # last_death_of_component = np.sort( pts0[:,1] )[::-1][2]
    # print intersection_homs, last_death_of_component
    # if intersection_homs[1] > last_death_of_component:
    #     print "Filter value didnt go through"
    #     if last_death_of_component > intersection_homs[0]:
    #         filt_value = (intersection_homs[1] - last_death_of_component) / 100.0 + last_death_of_component
    #     else:
    #         filt_value = (intersection_homs[1] - intersection_homs[0]) / 100.0 + intersection_homs[0]
    #     _tetrahedra = []
    #     for s in f:
    #         if s.data <= filt_value:
    #             if s.dimension() == 3:
    #                 _tetrahedra.append([s[0], s[1], s[2], s[3]])
    #             if s.dimension() == 2:
    #                 _tetrahedra.append([s[0], s[1], s[2]])
    #             if s.dimension() == 1:
    #                 _tetrahedra.append([s[0], s[1]])
    #
    #     save_to_obj(axes, _tetrahedra, path + 'filtration_{}.obj'.format(name))
    #print "Plotting and extracting surface took {} seconds".format(time.time() - start_extract)

def plot_and_save_info_2D(functionforward, diagramlayer, P, W, faces, centers, points, dimension, entropy, probability, t_kthdistance, name="", factor=(-1.0), path='../../results_reconstruct/'):
    # with torch.no_grad():
    print "#### Plot Save 2D"
    plt.figure()
    plt.triplot(points[:,0], points[:,1], faces)
    plt.savefig(path + 'grid_{}.png'.format(name))
    np.savetxt(path + 'grid_{}.txt'.format(name), points, delimiter=",")

    plt.figure()
    plt.scatter(centers[:,0], centers[:,1])
    plt.savefig(path + 'centers_scatter_{}.png'.format(name))

    func = functionforward(W, centers, points, dimension, entropy, probability, t_kthdistance)

    dgms = diagramlayer(func, P)
    dgms = dgms.numpy()
    np.savetxt(path + 'weights_{}.txt'.format(name), W, delimiter=',')

    func = func.numpy()
    np.savetxt(path + "function_{}.txt".format(name), func, delimiter=",")
    plt.figure()
    plt.scatter(points[:,0], points[:,1], c=func.flatten())
    plt.colorbar()
    plt.savefig(path + 'function_values_{}.png'.format(name))

    dgmpts = dgms[0]
    dgmpts = np.delete(dgmpts, np.where((dgmpts == (-np.inf, -np.inf)).all(axis=1)), axis=0)
    if len(dgmpts) > 0:
        fig = plot_diagram2(dgmpts, 'Dimension {}'.format(0))
    else:
        fig = plt.figure()
    fig.savefig(path + 'dgm0_{}.png'.format(name))

    dgmpts = dgms[1]
    dgmpts = np.delete(dgmpts, np.where((dgmpts == (-np.inf, -np.inf)).all(axis=1)), axis=0)
    #print dgmpts
    if len(dgmpts[dgmpts > -np.inf]) == 0:

        print "#### WARNING: No dmg-1 pts"

    if len(dgmpts) > 0:
        plot_points = dgmpts
        fig = plot_diagram2(plot_points, 'Dimension {}'.format(1))
        fig.savefig(path + 'dgm1_{}.png'.format(name))

        life_times = np.abs(plot_points[:,1] - plot_points[:,0])
        desc_sort = np.argsort(life_times)[::-1]
        #print "top", life_times[desc_sort]
        plt.figure()
        loop = np.zeros(func.shape)
        if factor < 0:
            loop[func >= plot_points[desc_sort[0], 0]] = 2
        else:
            loop[func >= plot_points[desc_sort[0], 0]] = 2

        plt.scatter(points[np.nonzero(loop)[0],0], points[np.nonzero(loop)[0],1])
        #imgplot = plt.imshow(loop.reshape(int(np.sqrt(loop.shape[0])), int(np.sqrt(loop.shape[0]))))
        plt.savefig(path + 'most_persistent_loop_{}.png'.format(name))

        if len(desc_sort) > 1:
            plt.figure()
            loop = np.zeros(func.shape)
            if factor < 0:
                loop[func >= plot_points[desc_sort[1], 0]] = 1
            else:
                loop[func <= plot_points[desc_sort[1], 0]] = 1
            #imgplot = plt.imshow(loop.reshape(int(np.sqrt(loop.shape[0])), int(np.sqrt(loop.shape[0]))))
            plt.scatter(points[np.nonzero(loop)[0],0], points[np.nonzero(loop)[0],1])
            plt.savefig(path + '2nd_most_persistent_loop_{}.png'.format(name))
    else:
        fig = plt.figure()
        fig.savefig(path + 'dgm1_{}.png'.format(name))

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
