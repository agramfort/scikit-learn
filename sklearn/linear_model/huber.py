from math import exp

import numpy as np

from scipy import optimize

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model.base import center_data, LinearModel
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y, check_array, check_consistent_length, column_or_1d

def _huber_loss_and_gradient(w, X, y, epsilon, alpha):
    """
    Calculate the robust huber loss as described in
    "A robust hybrid of lasso and ridge regression.

    """
    n_samples, n_features = X.shape
    fit_intercept = n_features + 2 == w.shape[0]
    if fit_intercept:
        intercept = w[-2]
    sigma = w[-1]
    w = w[:X.shape[1]]

    # Calculate the values where |y - X'w -c / exp(sigma)| > epsilon
    # The values above this threshold are outliers.
    linear_loss = y - np.dot(X, w)
    if fit_intercept:
        linear_loss -= intercept
    abs_linear_loss = np.abs(linear_loss)
    outliers_true = abs_linear_loss * exp(-sigma) > epsilon

    # Calculate the linear loss due to the outliers.
    # This is equal to (2 * M * |y - X'w / exp(sigma)| - M**2)*exp(sigma)
    n_outliers = np.count_nonzero(outliers_true)
    outliers = abs_linear_loss[outliers_true]
    outlier_loss = 2 * epsilon * np.sum(outliers) - exp(sigma) * n_outliers * epsilon**2

    # Calculate the quadratic loss due to the non-outliers.-
    # This is equal to |(y - X'w)**2 / exp(2*sigma)|*exp(sigma)
    non_outliers = linear_loss[~outliers_true]
    squared_loss = exp(-sigma) * np.dot(non_outliers, non_outliers)

    if fit_intercept:
        grad = np.zeros(n_features + 2)
    else:
        grad = np.zeros(n_features + 1)

    # Gradient due to the squared loss.
    grad[:n_features] = 2 * exp(-sigma) * np.dot(non_outliers, -X[~outliers_true, :])

    # Gradient due to the linear loss.
    outliers_true_pos = np.logical_and(linear_loss >= 0, outliers_true)
    outliers_true_neg = np.logical_and(linear_loss < 0, outliers_true)
    grad[:n_features] -= 2 * epsilon * X[outliers_true_pos, :].sum(axis=0)
    grad[:n_features] += 2 * epsilon * X[outliers_true_neg, :].sum(axis=0)

    # Gradient due to the penalty.
    grad[:n_features] += alpha * 2 * w

    # Gradient due to sigma.
    grad[-1] = n_samples * exp(sigma)
    grad[-1] -= n_outliers * epsilon**2 * exp(sigma)
    grad[-1] -= squared_loss

    # Gradient due to the intercept.
    if fit_intercept:
        grad[-2] = -2 * exp(-sigma) * np.sum(non_outliers)
        grad[-2] -= 2 * epsilon * np.sum(outliers_true_pos)
        grad[-2] += 2 * epsilon * np.sum(outliers_true_neg)
    return X.shape[0] * exp(sigma) + squared_loss + outlier_loss + alpha * np.dot(w, w), grad


class HuberRegressor(LinearModel, RegressorMixin, BaseEstimator):
    def __init__(self, epsilon=1.35, n_iter=100, alpha=0.0001,
                 warm_start=False, copy=True, fit_intercept=True):
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.alpha = alpha
        self.warm_start = warm_start
        self.copy = copy
        self.fit_intercept = fit_intercept

    def fit(self, X, y):

        # We can use check_X_y directly after consensus on
        # https://github.com/scikit-learn/scikit-learn/pull/5312 is reached.
        X = check_array(X, copy=self.copy)
        y = column_or_1d(y, warn=True)
        check_consistent_length(X, y)

        coef = getattr(self, 'coef_', None)
        if not self.warm_start or (self.warm_start and coef is None):
            if self.fit_intercept:
                self.coef_ = np.zeros(X.shape[1] + 2)
            else:
                self.coef_ = np.zeros(X.shape[1] + 1)

        try:
            self.coef_, f, self.dict_ = optimize.fmin_l_bfgs_b(
                _huber_loss_and_gradient, self.coef_, approx_grad=True,
                args=(X, y, self.epsilon, self.alpha), maxiter=self.n_iter, pgtol=1e-3)
        except TypeError:
            self.coef_, f, self.dict_ = optimize.fmin_l_bfgs_b(
                _huber_loss_and_gradient, self.coef_,
                args=(X, y, self.epsilon, self.alpha))

        self.scale_ = self.coef_[-1]
        if self.fit_intercept:
            self.intercept_ = self.coef_[-2]
        else:
            self.intercept_ = 0.0
        self.coef_ = self.coef_[:X.shape[1]]

        self.loss_ = f
        return self
