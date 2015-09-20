import numpy as np

from scipy import optimize

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model.base import center_data
from sklearn.utils import check_X_y

def _huber_loss_and_gradient(w, X, y, epsilon, alpha):

    linear_loss = y - np.dot(X, w)
    abs_linear_loss = np.abs(linear_loss)
    outliers_true = abs_linear_loss > epsilon
    n_outliers = np.count_nonzero(outliers_true)
    outliers = linear_loss[outliers_true]
    outlier_loss = epsilon * np.sum(outliers) - n_outliers * 0.5 * epsilon**2
    non_outliers = linear_loss[~outliers_true]
    loss = 0.5 * np.dot(non_outliers, non_outliers) + outlier_loss

    # Calulate the gradient
    grad = np.zeros(w.shape[0])

    return loss + alpha * np.dot(w, w)


class HuberRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, epsilon=0.1, n_iter=100, alpha=0.0001):
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.alpha = alpha

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        coef = np.zeros(X.shape[0])
        coef, f, _ = optimize.fmin_l_bfgs_b(
        	_huber_loss_and_gradient, coef, approx_grad=True,
        	args=(X, y, self.epsilon, self.alpha), maxiter=self.n_iter)
        self.coef_ = coef
        self.f_ = f
        return self
