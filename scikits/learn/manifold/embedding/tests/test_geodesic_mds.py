#!/usr/bin/env python

import unittest
import numpy
from numpy.testing import assert_array_equal, \
                          assert_array_almost_equal, \
                          assert_raises

from unittest import TestCase
from ....neighbors import Neighbors
from ..geodesic_mds import populate_distance_matrix_from_neighbors

class TestPopulateMatrix(TestCase):
  def test_main(self):
    samples = numpy.array((0., 0., 0.,
      1., 0., 0.,
      0., 1., 0.,
      1., 1., 0.,
      0., .5, 0.,
      .5, 0., 0.,
      1., 1., 0.5,
      )).reshape((-1,3))
    neigh = Neighbors(k=3)
    neigh.fit(samples)
    neigh = neigh.kneighbors
    dists = populate_distance_matrix_from_neighbors(samples, neigh)
    assert_array_equal(dists, dists.T)

if __name__ == "__main__":
  unittest.main()
  