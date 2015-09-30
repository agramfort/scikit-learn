import numpy as np
from scipy import optimize

from sklearn.utils.testing import assert_equal, assert_almost_equal

from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.linear_model.huber import _huber_loss_and_gradient

X, y = make_regression(n_samples=50, n_features=20, random_state=0)
rng = np.random.RandomState(0)

def test_huber_equals_ridge_for_high_epsilon():
    ridge = Ridge(fit_intercept=True, random_state=0, alpha=0.0)
    ridge.fit(X, y)
    huber = HuberRegressor(fit_intercept=True, epsilon=10000.0, alpha=0.0)
    huber.fit(X, y)
    assert_almost_equal(huber.coef_, ridge.coef_, 4)


def test_huber_gradient():
    """Test that the gradient calculated by _huber_loss_and_gradient is correct"""

    loss_func = lambda x, *args: _huber_loss_and_gradient(x, *args)[0]
    grad_func = lambda x, *args: _huber_loss_and_gradient(x, *args)[1]

    # Check using optimize.check_grad that the gradients are equal.
    for i in range(5):

        for n_features in [X.shape[1] + 1, X.shape[1] + 2]:
            w = rng.randn(n_features)
            grad_same = optimize.check_grad(
                loss_func, grad_func, w, X, y, 0.01, 0.1)
            assert_almost_equal(grad_same, 1e-6, 4)
