import numpy as np
import sys
import torch

import sklearn
from sklearn.neighbors import NearestNeighbors

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

def knn(matrix, k): # distance between x and k nearest neighbor, takes top p proportion
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(matrix)
    distances, indices = nbrs.kneighbors(matrix) # last index is kth nearest
    kthdistances = distances[:, k-1] # last column
    return kthdistances

def try_range(start, stop, partitions, loss_fn):
    print "#### Trying range"
    best_param = start
    best_value = np.inf
    step = np.abs(stop - start)/partitions
    list = np.arange(start, stop, step)
    for i in range(len(list)):
        param_try = list[i]
        value_try = loss_fn(param_try)
        print "{} param: {} val: {}".format(i, param_try, value_try),
        if value_try == -np.inf: # OBS A way to return if found
            return param_try
        if value_try < best_value:
            best_value = value_try
            best_param = param_try
    return best_param

def get_points(centers, partitions, greatest_dist, dimension=3, denominator=50):
    points = centers
    coor_pts = points[:,0]
    abs_diff = np.abs( np.amin(coor_pts) - np.amax(coor_pts) )
    start_x = np.amin(coor_pts) - np.float(2 * greatest_dist)/denominator
    stop_x = np.amax(coor_pts) + np.float(2 * greatest_dist)/denominator
    coor_pts = points[:,1]
    abs_diff = np.abs( np.amin(coor_pts) - np.amax(coor_pts) )
    start_y = np.amin(coor_pts) - np.float(2 * greatest_dist)/denominator
    stop_y = np.amax(coor_pts) + np.float(2 * greatest_dist)/denominator

    axis_x = np.arange(start_x, stop_x, np.abs(stop_x - start_x)/partitions)
    axis_y = np.arange(start_y, stop_y, np.abs(stop_y - start_y)/partitions)

    if dimension > 2:
        coor_pts = points[:,2]
        abs_diff = np.abs( np.amin(coor_pts) - np.amax(coor_pts) )
        start_z = np.amin(coor_pts) - np.float(2 * greatest_dist)/denominator
        stop_z = np.amax(coor_pts) + np.float(2 * greatest_dist)/denominator
        axis_z = np.arange(start_z, stop_z, np.abs(stop_z - start_z)/partitions)
        grid_axes = np.array(np.meshgrid(axis_x, axis_y, axis_z)) # (3, 31, 31, 31)
        grid_axes = np.transpose(grid_axes, (1, 2, 3, 0))
    else:
        grid_axes = np.array(np.meshgrid(axis_x, axis_y)) # (3, 31, 31, 31)
        grid_axes = np.transpose(grid_axes, (1, 2, 0))

    axes = grid_axes.reshape([-1, dimension])
    return axes

# NVertices  NFaces  NEdges
# x[0]  y[0]  z[0]
# Nv = # vertices on this face
# v[0] ... v[Nv-1]: vertex indices in range 0..NVertices-1
def save_to_off(points, faces, path="./temp.off"):
    f = open(path, "w")
    f.write("OFF\n")
    f.write("{} {} {}\n".format(len(points), len(faces), 0))
    dim = points.shape[1]
    if dim == 3:
        for r in range(points.shape[0]):
            row = points[r]
            f.write("{} {} {}\n".format(row[0], row[1], row[2]))
    if dim == 2:
        for r in range(points.shape[0]):
            row = points[r]
            f.write("{} {}\n".format(row[0], row[1]))

    for fi in range(len(faces)):
        face = faces[fi]
        # if len(face) == 2:
        #     f.write("l {} {}\n".format(face[0] + 1, face[1] + 1))
        # elif len(face) == 3:
        #     f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))
        if len(face) == 3:
            f.write("3 {} {} {}\n".format(face[0], face[1], face[2]))
        elif len(face) == 4:
            f.write("4 {} {} {} {}\n".format(face[0], face[1], face[2], face[3]))
    f.close()
    return path
