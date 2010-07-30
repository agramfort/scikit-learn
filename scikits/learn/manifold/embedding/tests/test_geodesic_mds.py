#!/usr/bin/env python

import unittest
import numpy
from numpy.testing import assert_array_equal, \
                          assert_array_almost_equal, \
                          assert_raises

from unittest import TestCase
from tempfile import NamedTemporaryFile
from nose.tools import raises

from ..tools import create_neighborer, dist2hd
from ..geodesic_mds import reduct, populate_distance_matrix_from_neighbors, \
    Isomap

samples = numpy.array((0., 0., 0.,
  1., 0., 0.,
  0., 1., 0.,
  1., 1., 0.,
  0., .5, 0.,
  .5, 0., 0.,
  1., 1., 0.5,
  )).reshape((-1,3))

distances = numpy.array(((0, 1, 1, 2, .5, .5, 2.11803399),
         (1, 0, 2, 3, 1.5, .5, 3.11803399),
         (1, 2, 0, 1, .5, 1.5, 1.11803399),
         (2, 3, 1, 0, 1.5, 2.5, .5),
         (.5, 1.5, .5, 1.5, 0, 1, 1.61803399),
         (.5, .5, 1.5, 2.5, 1, 0, 2.61803399),
         (2.11803399, 3.11803399, 1.11803399, .5, 1.61803399, 2.61803399, 0)))

def reduction(dists, function, n_coords):
    assert(n_coords == 2)
    assert(dists.shape == (7, 7))
    assert_array_almost_equal(distances, numpy.asarray(dists), decimal=2)
                         
class TestPopulateMatrix(TestCase):
    def test_main(self):
        neigh = create_neighborer(samples, n_neighbors=3)
        dists = populate_distance_matrix_from_neighbors(samples, neigh)
        assert_array_equal(dists, dists.T)

class TestReduct(TestCase):
    def test_simple_reduct(self):
        reduct(reduction, None, samples, 2, None, 3, None, None)

    def test_cached_reduct(self):
        temp = NamedTemporaryFile()
        numpy.save(temp.file, distances)
        temp.file.flush()
        temp.file.seek(0)
        reduct(reduction, None, samples, 2, None, 3, None, temp.file)

    def test_to_be_cached_reduct(self):
        temp = NamedTemporaryFile()
        reduct(reduction, None, samples, 2, None, 3, None, temp.file)
        temp.file.flush()
        temp.file.seek(0)
        dists = numpy.load(temp.file)
        assert_array_almost_equal(distances, dists)

class TestIsomap(TestCase):
    def test_fit(self):
        isomap = Isomap(n_coords = 2, mapping_kind = None, n_neighbors = 3)
        assert(isomap.fit(samples[:3]) == isomap)
        assert(hasattr(isomap, 'embedding_'))
        assert(isomap.embedding_.shape == (3, 2))
        assert_array_almost_equal(dist2hd(isomap.embedding_[:3], 
            isomap.embedding_[:3])**2, distances[:3, :3])

    @raises(RuntimeError)
    def test_transform_raises(self):
        isomap = Isomap(n_coords = 2, mapping_kind = None, n_neighbors = 3)
        isomap.fit(samples[:3])
        isomap.transform(samples[0])

    def test_transform(self):
        isomap = Isomap(n_coords = 2, n_neighbors = 3)
        isomap.fit(samples[:3])
        mapped = isomap.transform(samples)
        assert_array_almost_equal(mapped[:3], isomap.embedding_, decimal=3)
        
if __name__ == "__main__":
  unittest.main()
  