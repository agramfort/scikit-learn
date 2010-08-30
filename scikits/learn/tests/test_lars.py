from scikits.learn import lars, datasets

import numpy as np
from numpy.testing import *


n, m = 10, 10
np.random.seed (0)
diabetes = datasets.load_diabetes()
X, Y = diabetes.data, diabetes.target


#normalize data
_xmean = X.mean(0)
_ymean = Y.mean(0)
X = X - _xmean
Y = Y - _ymean
_norms = np.apply_along_axis (np.linalg.norm, 0, X)
nonzeros = np.flatnonzero(_norms)
X[:, nonzeros] /= _norms[nonzeros]


def test_1():
    """
    Principle of LARS is to keep covariances tied and decreasing
    """
    clf = lars.LassoLARS()
    clf.fit (X, Y, max_features=6)
    for (i, coef_) in enumerate(clf.coef_path_.T):
        res =  Y - np.dot(X, coef_)
        cov = np.dot (X.T, res)
        C = np.max (abs(cov))
        eps = 1e-3
        ocur = len(cov[ C - eps < abs(cov)])
        assert ocur == i + 1
