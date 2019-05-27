# utilities to help construction of complexes

from topologylayer.functional.cohom_cpp import SimplicialComplex
from itertools import combinations
import numpy as np


def unique_simplices(faces, dim):
    """
    obtain unique simplices up to dimension dim from faces
    """
    simplices = [[] for k in range(dim+1)]
    # loop over faces
    for face in faces:
        # loop over dimension
        for k in range(dim+1):
            # loop over simplices
            for s in combinations(face, k+1):
                simplices[k].append(np.sort(list(s)))

    s = SimplicialComplex()
    # loop over dimension
    for k in range(dim+1):
        kcells = np.unique(simplices[k], axis=0)
        for cell in kcells:
            s.append(cell)

    return s
