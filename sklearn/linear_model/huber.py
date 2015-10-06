from math import exp

import numpy as np

from scipy import optimize

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model.base import center_data, LinearModel
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y, check_array, check_consistent_length, column_or_1d

def _huber_loss_and_gradient(w, X, y, epsilon, alpha, sample_weight=None):
    """
    Returns the huber loss and the gradient.

    Parameters
    ----------
    w: ndarray, shape (n_features + 1,) or (n_features + 2,)
        Feature vector.
        w[:n_features] gives the feature vector
        w[-1] gives the scale factor and if the intercept is fit w[-2]
        gives the intercept factor.

    X: ndarray, shape (n_samples, n_features)
        Input data.

    y: ndarray, shape (n_samples,)
        Target vector.

    epsilon: float
        Measures the robustness of the huber estimator.

    alpha: float
        Regularization parameter.

    Returns
    -------
    loss: float
        Huber loss.

    gradient: ndarray, shape (n_features + 1,) or (n_features + 2,)
        Returns the derivative of the huber loss with respect to each
        coefficient, intercept and the scale as a vector.
    """
    n_samples, n_features = X.shape
    fit_intercept = n_features + 2 == w.shape[0]
    if fit_intercept:
        intercept = w[-2]
    sigma = w[-1]
    w = w[:X.shape[1]]

    if sample_weight is not None:
        n_samples = np.sum(sample_weight)

    # Calculate the values where |y - X'w -c / exp(sigma)| > epsilon
    # The values above this threshold are outliers.
    linear_loss = y - np.dot(X, w)
    if fit_intercept:
        linear_loss -= intercept
    abs_linear_loss = np.abs(linear_loss)
    outliers_true = abs_linear_loss * exp(-sigma) > epsilon

    # Calculate the linear loss due to the outliers.
    # This is equal to (2 * M * |y - X'w / exp(sigma)| - M**2)*exp(sigma)
    outliers = abs_linear_loss[outliers_true]
    if sample_weight is None:
        n_outliers = np.count_nonzero(outliers_true)
        outlier_loss = (
            2 * epsilon * np.sum(outliers) -
            exp(sigma) * n_outliers * epsilon**2)
    else:
        outliers_sw = sample_weight[outliers_true]
        n_outliers = np.sum(outliers_sw)
        outlier_loss = (
            2 * epsilon * np.sum(outliers_sw * outliers) -
            exp(sigma) * n_outliers * epsilon**2)


    # Calculate the quadratic loss due to the non-outliers.-
    # This is equal to |(y - X'w)**2 / exp(2*sigma)|*exp(sigma)
    non_outliers = linear_loss[~outliers_true]
    if sample_weight is None:
        squared_loss = exp(-sigma) * np.dot(non_outliers, non_outliers)
    else:
        weighted_non_outliers = sample_weight[~outliers_true] * non_outliers
        weighted_loss = np.dot(weighted_non_outliers, non_outliers)
        squared_loss = exp(-sigma) * weighted_loss

    if fit_intercept:
        grad = np.zeros(n_features + 2)
    else:
        grad = np.zeros(n_features + 1)

    # Gradient due to the squared loss.
    if sample_weight is None:
        grad[:n_features] = (
            2 * exp(-sigma) * np.dot(non_outliers, -X[~outliers_true, :]))
    else:
        grad[:n_features] = (
            2 * exp(-sigma) * np.dot(weighted_non_outliers, -X[~outliers_true, :])
        )

    # Gradient due to the linear loss.
    outliers_true_pos = np.logical_and(linear_loss >= 0, outliers_true)
    outliers_true_neg = np.logical_and(linear_loss < 0, outliers_true)
    if sample_weight is None:
        grad[:n_features] -= 2 * epsilon * X[outliers_true_pos, :].sum(axis=0)
        grad[:n_features] += 2 * epsilon * X[outliers_true_neg, :].sum(axis=0)
    else:
        sample_weight_outliers_true_pos = sample_weight[outliers_true_pos].reshape(-1, 1)
        sample_weight_outliers_true_neg = sample_weight[outliers_true_neg].reshape(-1, 1)
        grad[:n_features] -= 2 * epsilon * (
            (sample_weight_outliers_true_pos * X[outliers_true_pos, :]).sum(axis=0))
        grad[:n_features] += 2 * epsilon * (
            (sample_weight_outliers_true_neg * X[outliers_true_neg, :]).sum(axis=0))

    # Gradient due to the penalty.
    grad[:n_features] += alpha * 2 * w

    # Gradient due to sigma.
    grad[-1] = n_samples * exp(sigma)
    grad[-1] -= n_outliers * epsilon**2 * exp(sigma)
    grad[-1] -= squared_loss

    # Gradient due to the intercept.
    if fit_intercept:
        if sample_weight is None:
            grad[-2] = -2 * exp(-sigma) * np.sum(non_outliers)
            grad[-2] -= 2 * epsilon * np.sum(outliers_true_pos)
            grad[-2] += 2 * epsilon * np.sum(outliers_true_neg)
        else:
            grad[-2] = -2 * exp(-sigma) * np.sum(weighted_non_outliers)
            grad[-2] -= 2 * epsilon * np.sum(sample_weight[outliers_true_pos])
            grad[-2] += 2 * epsilon * np.sum(sample_weight[outliers_true_neg])

    return n_samples * exp(sigma) + squared_loss + outlier_loss + alpha * np.dot(w, w), grad


class HuberRegressor(LinearModel, RegressorMixin, BaseEstimator):
    """
    Linear regression model that is robust to outliers.

    The Huber Regressor optimizes the squared loss for the samples where
    |y - X'w / exp(sigma)| > epsilon and the linear loss for the samples
    where |y - X'w / exp(sigma)| < epsilon, where w and sigma are parameters
    to be optimized. The parameter sigma makes sure that if y is scaled up
    or down by a certain factor, one does not need to rescale M to acheive
    the same robustness.

    This makes sure that the loss function is not heavily influenced by the
    outliers while not completely ignoring their effect.

    More specifically the loss function optimized is.
        X.shape[0] * exp(sigma) + (
            \sum_{i=1}^{i=X.shape[0]}H(y_{i} - X_{i}*w - c / exp(sigma))
            * exp(sigma))
    where

        - H(z) = z**2 when |z| < epsilon
        - H(z) = 2 * epsilon * |z| - epsilon ** 2 when |z| > epsilon

    This loss function is jointly convex in sigma and w and hence can be
    optimized together.

    Parameters
    ----------
    epsilon: float, greater than 1.0, default 1.35
        The parameter epsilon controls the number of samples that should be
        classified as outliers. The lesser the epsilon, the more robust it is
        to outliers.

    n_iter: int, default 100
        Number of iterations that scipy.optimize.fmin_l_bfgs_b should run for.

    alpha: float, default 0.0001
        Regularization parameter.

    warm_start: bool, default False
        If warm_start is set to False, the coefficients will be overwritten
        for every call to fit.

    fit_intercept: bool, default True
        Whether or not to fit the intercept. This can be set to False for
        if the data is already centered around the origin.

    Attributes
    ----------
    coef_: array, shape (n_features,)
        Features got by optimizing the huber loss.

    intercept_: float
        Bias factor.

    scale_ : float
        The value by which |y - X'w -c| is scaled down by.

    References
    ----------
    Art B. Owen (2006), A robust hybrid of lasso and ridge regression.
    http://statweb.stanford.edu/~owen/reports/hhu.pdf

    Notes
    -----
    Though, the paper says to use sigma, we use exp(sigma). This removes the
    constraint that sigma should be positive.
    """

    def __init__(self, epsilon=1.35, n_iter=100, alpha=0.0001,
                 warm_start=False, fit_intercept=True):
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.alpha = alpha
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """
        # We can use check_X_y directly after consensus on
        # https://github.com/scikit-learn/scikit-learn/pull/5312 is reached.
        X = check_array(X, copy=False)
        y = column_or_1d(y, warn=True)
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
            check_consistent_length(X, y, sample_weight)
        else:
            check_consistent_length(X, y)

        if self.epsilon < 1.0:
            raise ValueError(
                "epsilon should be greater than 1.0, got %f" % self.epsilon)
        coef = getattr(self, 'coef_', None)
        if not self.warm_start or (self.warm_start and coef is None):
            if self.fit_intercept:
                self.coef_ = np.zeros(X.shape[1] + 2)
            else:
                self.coef_ = np.zeros(X.shape[1] + 1)

        try:
            self.coef_, f, self.dict_ = optimize.fmin_l_bfgs_b(
                _huber_loss_and_gradient, self.coef_, approx_grad=True,
                args=(X, y, self.epsilon, self.alpha, sample_weight),
                maxiter=self.n_iter, pgtol=1e-3)
        except TypeError:
            self.coef_, f, self.dict_ = optimize.fmin_l_bfgs_b(
                _huber_loss_and_gradient, self.coef_,
                args=(X, y, self.epsilon, self.alpha, sample_weight))

        self.scale_ = self.coef_[-1]
        if self.fit_intercept:
            self.intercept_ = self.coef_[-2]
        else:
            self.intercept_ = 0.0
        self.coef_ = self.coef_[:X.shape[1]]

        self.loss_ = f
        return self
