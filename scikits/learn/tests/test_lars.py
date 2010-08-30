from scikits.learn import lars, datasets
from scikits.learn import glm

import numpy as np
# from numpy.testing import *


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
    
    alphas_, active, coef_path_ = lars.lars (X, Y, 6, method="lar")
    for (i, coef_) in enumerate(coef_path_.T):
        res =  Y - np.dot(X, coef_)
        cov = np.dot (X.T, res)
        C = np.max (abs(cov))
        eps = 1e-3
        ocur = len(cov[ C - eps < abs(cov)])
        assert ocur == i + 1

def test_lasso_lars_vs_lasso_cd():
    """
    Test that LassoLars and Lasso using coordinate descent give the
    same results
    """
    lasso_lars = lars.LassoLARS(alpha=0.1)
    lasso_lars.fit(X, Y)

    # make sure results are the same than with Lasso Coordinate descent
    lasso = glm.Lasso(alpha=0.1)
    lasso.fit(X, Y, maxit=3000, tol=1e-10)

    error = np.linalg.norm(lasso_lars.coef_ - lasso.coef_)
    assert error < 1e-5
