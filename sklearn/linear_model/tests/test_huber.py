# Authors: Manoj Kumar mks542@nyu.edu
# License: BSD 3 clause

import numpy as np
from scipy import optimize, sparse

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_greater

from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, Ridge, SGDRegressor
from sklearn.linear_model.huber import _huber_loss_and_gradient

rng = np.random.RandomState(0)


def make_regression_with_outliers(n_samples=50, n_features=20):
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        random_state=0, noise=10.0)

    # Replace 10% of the sample with noise.
    num_noise = int(0.1 * n_samples)
    random_samples = rng.randint(0, n_samples, num_noise)
    X[random_samples, :] = rng.normal(0, 1, (num_noise, X.shape[1]))
    return X, y


def test_huber_equals_ridge_for_high_epsilon():
    """Test that Ridge matches HuberRegressor for large epsilon"""
    X, y = make_regression_with_outliers()
    ridge = Ridge(fit_intercept=True, alpha=0.0, solver='svd')
    ridge.fit(X, y)
    huber = HuberRegressor(fit_intercept=True, epsilon=1e5, alpha=0.0)
    huber.fit(X, y)
    assert_almost_equal(huber.coef_, ridge.coef_, 3)
    assert_almost_equal(huber.intercept_, ridge.intercept_, 3)


def test_huber_gradient():
    """Test that the gradient calculated by _huber_loss_and_gradient is ok"""

    X, y = make_regression_with_outliers()
    loss_func = lambda x, *args: _huber_loss_and_gradient(x, *args)[0]
    grad_func = lambda x, *args: _huber_loss_and_gradient(x, *args)[1]

    # Check using optimize.check_grad that the gradients are equal.
    for _ in range(5):
        # Check for both fit_intercept and otherwise.
        for n_features in [X.shape[1] + 1, X.shape[1] + 2]:
            w = rng.randn(n_features)
            w[-1] = np.abs(w[-1])
            grad_same = optimize.check_grad(
                loss_func, grad_func, w, X, y, 0.01, 0.1)
            assert_almost_equal(grad_same, 1e-6, 4)


def test_huber_sample_weights():
    """Test sample_weights implementation in HuberRegressor"""

    X, y = make_regression_with_outliers()
    huber = HuberRegressor(fit_intercept=True, alpha=0.1)
    huber.fit(X, y)
    huber_coef = huber.coef_
    huber_intercept = huber.intercept_

    huber.fit(X, y, sample_weight=np.ones(y.shape[0]))
    assert_array_almost_equal(huber.coef_, huber_coef)
    assert_array_almost_equal(huber.intercept_, huber_intercept)

    X, y = make_regression_with_outliers(n_samples=5, n_features=20)
    X[X < 0.3] = 0.0
    X_new = np.vstack((X, np.vstack((X[1], X[1], X[3]))))
    y_new = np.concatenate((y, [y[1]], [y[1]], [y[3]]))
    huber.fit(X_new, y_new)
    huber_coef = huber.coef_
    huber_intercept = huber.intercept_
    huber.fit(X, y, sample_weight=[1, 3, 1, 2, 1])
    assert_array_almost_equal(huber.coef_, huber_coef)
    assert_array_almost_equal(huber.intercept_, huber_intercept)

    # Test sparse implementation with sparse weights.
    # Checking sparse=non_sparse should be covered in the common tests.
    X_csr = sparse.csr_matrix(X)
    huber_sparse = HuberRegressor(fit_intercept=True, alpha=0.1)
    huber_sparse.fit(X_csr, y, sample_weight=[1, 3, 1, 2, 1])
    assert_array_almost_equal(huber_sparse.coef_, huber_coef)


def return_number_outliers(X, y, coef, intercept, scale, epsilon):
    """Return the number of outliers."""
    outliers = np.abs(np.dot(X, coef) + intercept - y) > epsilon * scale
    return np.sum(outliers)


def test_huber_scaling_invariant():
    """Test that outliers filtering is scaling independent."""
    X, y = make_regression_with_outliers()
    huber = HuberRegressor(fit_intercept=False, alpha=0.0, n_iter=100,
                           epsilon=1.35)
    huber.fit(X, y)
    n_outliers1 = return_number_outliers(
        X, y, huber.coef_, huber.intercept_, huber.scale_, huber.epsilon)

    huber.fit(X, 2. * y)
    n_outliers2 = return_number_outliers(
        X, 2. * y, huber.coef_, huber.intercept_, huber.scale_, huber.epsilon)

    huber.fit(2. * X, 2. * y)
    n_outliers3 = return_number_outliers(
        2. * X, 2. * y, huber.coef_, huber.intercept_, huber.scale_,
        huber.epsilon)

    random_features = rng.randint(0, 20, (5,))
    noise = rng.normal(0, 1, 5)
    X_new = np.copy(X)
    X_new[:, random_features] /= noise
    huber.fit(X_new, y)
    n_outliers4 = return_number_outliers(
        X_new, y, huber.coef_, huber.intercept_, huber.scale_, huber.epsilon)

    assert_equal(n_outliers1, n_outliers2)
    assert_equal(n_outliers3, n_outliers4)
    assert_equal(n_outliers1, n_outliers4)


def test_huber_and_sgd_same_results():
    """Test they should converge to same coefficients for same parameters"""

    X, y = make_regression_with_outliers(n_samples=5, n_features=1)

    # Fit once to find out the scale parameter. Scale down X and y by scale
    # so that the scale parameter is optimized to 1.0
    huber = HuberRegressor(fit_intercept=False, alpha=0.0, n_iter=100,
                           epsilon=1.35)
    huber.fit(X, y)
    X_scale = X / huber.scale_
    y_scale = y / huber.scale_
    huber.fit(X_scale, y_scale)
    assert_almost_equal(huber.scale_, 1.0, 3)

    sgdreg = SGDRegressor(
        alpha=0.0, loss="huber", shuffle=True, random_state=0, n_iter=1000000,
        fit_intercept=False, epsilon=1.35)
    sgdreg.fit(X_scale, y_scale)
    assert_array_almost_equal(huber.coef_, sgdreg.coef_, 2)


def test_huber_warm_start():
    X, y = make_regression_with_outliers()
    huber_warm = HuberRegressor(
        fit_intercept=True, alpha=1.0, n_iter=5, warm_start=True)
    huber_warm.fit(X, y)
    huber_warm.fit(X, y)

    huber_cold = HuberRegressor(fit_intercept=True, alpha=1.0, n_iter=5)
    huber_cold.fit(X, y)
    assert_almost_equal(huber_warm.coef_, huber_cold.coef_, 3)


def test_huber_better_r2_score():
    """Test that huber returns a better r2 score than non-outliers"""
    X, y = make_regression_with_outliers()
    huber = HuberRegressor(fit_intercept=True, alpha=0.01, n_iter=100)
    huber.fit(X, y)
    linear_loss = np.dot(X, huber.coef_) + huber.intercept_ - y
    mask = np.abs(linear_loss) < huber.epsilon * huber.scale_
    huber_score = huber.score(X[mask], y[mask])
    huber_outlier_score = huber.score(X[~mask], y[~mask])

    ridge = Ridge(fit_intercept=True, alpha=0.01)
    ridge.fit(X, y)
    ridge_score = ridge.score(X[mask], y[mask])
    ridge_outlier_score = ridge.score(X[~mask], y[~mask])
    assert_greater(huber_score, ridge_score)

    # The huber model should also fit poorly on the outliers.
    assert_greater(ridge_outlier_score, huber_outlier_score)
