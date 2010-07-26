#!/usr/bin/env python

import unittest
import numpy
from numpy.testing import assert_array_equal, \
                          assert_array_almost_equal, \
                          assert_raises

from unittest import TestCase
from ..distances import numpy_floyd

class TestNumpyFloyd(TestCase):
  def test_lower_distances(self):
    d = numpy.array(((0, 2, 3, 4), (1, 0, 3, 4), (1, 2, 0, 4), (1, 2, 3, 0)))
    d1 = d.copy()
    numpy_floyd(d1)
    assert((d1 <= d).all())
  
  def test_correct_estimation(self):
    d = numpy.array(((0, 1, 3, 3, 1), (1, 0, 1, 3, 3), (3, 1, 0, 1, 3), (2, 2, 1, 0, 1), (1, 2, 2, 1, 0)))
    d1 = d.copy()
    numpy_floyd(d1)
    d = numpy.array(((0, 1, 2, 2, 1), (1, 0, 1, 2, 2), (2, 1, 0, 1, 2), (2, 2, 1, 0, 1), (1, 2, 2, 1, 0)))
    assert_array_equal(d1, d)

if __name__ == "__main__":
  unittest.main()
  