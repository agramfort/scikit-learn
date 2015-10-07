from math import exp

import numpy as np
from scipy import optimize

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_almost_equal

from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.linear_model.huber import _huber_loss_and_gradient

X, y = make_regression(n_samples=50, n_features=20, random_state=0)
rng = np.random.RandomState(0)

# Replace 10% of the sample with outliers.
random_samples = rng.randint(0, 50, 5)
X_mean = np.mean(X, axis=0)
X[random_samples, :] = rng.normal(0, 1, (5, 20))

def test_huber_equals_ridge_for_high_epsilon():
    ridge = Ridge(fit_intercept=True, random_state=0, alpha=0.0)
    ridge.fit(X, y)
    huber = HuberRegressor(fit_intercept=True, epsilon=10000.0, alpha=0.0)
    huber.fit(X, y)
    assert_almost_equal(huber.coef_, ridge.coef_, 1)


def test_huber_gradient():
    """Test that the gradient calculated by _huber_loss_and_gradient is correct"""

    loss_func = lambda x, *args: _huber_loss_and_gradient(x, *args)[0]
    grad_func = lambda x, *args: _huber_loss_and_gradient(x, *args)[1]

    # Check using optimize.check_grad that the gradients are equal.
    for i in range(5):

        # Check for both fit_intercept and otherwise.
        for n_features in [X.shape[1] + 1, X.shape[1] + 2]:
            w = rng.randn(n_features)
            grad_same = optimize.check_grad(
                loss_func, grad_func, w, X, y, 0.01, 0.1)
            assert_almost_equal(grad_same, 1e-6, 4)


def test_huber_sample_weights():
    """Test sample_weights implementation in HuberRegressor"""

    X, y = make_regression(n_samples=50, n_features=20, random_state=0)
    huber = HuberRegressor(fit_intercept=True, alpha=0.1)
    huber.fit(X, y)
    huber_coef = huber.coef_
    huber_intercept = huber.intercept_

    huber.fit(X, y, sample_weight=np.ones(y.shape[0]))
    assert_array_almost_equal(huber.coef_, huber_coef)
    assert_array_almost_equal(huber.intercept_, huber_intercept)

    X, y = make_regression(n_samples=5, n_features=20, random_state=1)
    X_new = np.vstack((X, np.vstack((X[1], X[1], X[3]))))
    y_new = np.concatenate((y, [y[1]], [y[1]], [y[3]]))
    huber.fit(X_new, y_new)
    huber_coef = huber.coef_
    huber_intercept = huber.intercept_
    huber.fit(X, y, sample_weight=[1, 3, 1, 2, 1])
    assert_array_almost_equal(huber.coef_, huber_coef)
    assert_array_almost_equal(huber.intercept_, huber_intercept)


def return_number_outliers(X, y, coef, intercept, scale, epsilon):
    """Return the number of outliers."""
    outliers = np.abs(np.dot(X, coef) + intercept - y) > epsilon * exp(scale)
    return np.sum(outliers)


def test_huber_scaling_invariant():
    """Test that outliers filtering is scaling independent."""
    huber = HuberRegressor(fit_intercept=False, alpha=0.0, n_iter=100, epsilon=2.0)
    huber.fit(X, y)
    n_outliers = return_number_outliers(
        X, y, huber.coef_, huber.intercept_, huber.scale_, huber.epsilon)

    huber.fit(X, 2*y)
    n_outliers = return_number_outliers(
        X, 2*y, huber.coef_, huber.intercept_, huber.scale_, huber.epsilon)
    print(n_outliers)
