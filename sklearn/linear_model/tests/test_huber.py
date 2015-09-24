import numpy as np
from scipy import optimize

from sklearn.utils.testing import assert_equal, assert_almost_equal

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model.huber import _huber_loss_and_gradient

X, y = make_regression(n_samples=50, n_features=20)
scaler = StandardScaler(with_mean=True, with_std=True)
y = np.reshape(y, (-1, 1))
rng = np.random.RandomState(0)

X = scaler.fit_transform(X)
y = scaler.fit_transform(y).ravel()

def test_huber_loss():
    """Test that the loss calculated by _huber_loss_and_gradient is correct"""

    # When epsilon is large it should reduce to the squared loss
    w = rng.randn(X.shape[1])
    huber_loss = _huber_loss_and_gradient(w, X, y, 10000.0, 0.1)[0]
    squared_loss = 0.5 * np.sum((y - np.dot(X, w))**2) + 0.1 * np.dot(w, w)
    assert_equal(huber_loss, squared_loss, 4)

    # When epsilon is very small, this should reduce to the linear loss.
    epsilon = 0.01
    huber_loss = _huber_loss_and_gradient(w, X, y, 0.01, 0.1)[0]
    linear_loss = (epsilon * np.sum(np.abs(y - np.dot(X, w))) -
    	           X.shape[0] * epsilon * epsilon / 2 +
    	           0.1 * np.dot(w, w))
    assert_almost_equal(huber_loss, linear_loss, 4)


def test_huber_gradient():
    """Test that the gradient calculated by _huber_loss_and_gradient is correct"""

    loss_func = lambda x, *args: _huber_loss_and_gradient(x, *args)[0]
    grad_func = lambda x, *args: _huber_loss_and_gradient(x, *args)[1]

    # Check using optimize.check_grad that the gradients are equal.
    for i in range(5):
        w = rng.randn(X.shape[1])
        grad_same = optimize.check_grad(
            loss_func, grad_func, w, X, y, 0.01, 0.1)
        assert_almost_equal(grad_same, 1e-6, 4)


# def test_huber_convergence():
# 	"""Test that the loss function converges to zero."""
