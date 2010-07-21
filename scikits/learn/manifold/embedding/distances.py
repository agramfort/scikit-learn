
"""
Allows to compute the nearest neighbors
"""

import numpy
from .tools import dist2hd

def numpy_floyd(dists):
    """
    Implementation with Numpy vector operations
    """
    for indice1 in xrange(len(dists)):
        for indice2 in xrange(len(dists)):
            dists[indice2, :] = numpy.minimum(dists[indice2, :], dists[indice2, indice1] + dists[indice1, :])
