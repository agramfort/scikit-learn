from math import exp

import numpy as np

from scipy import optimize, sparse

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model.base import center_data, LinearModel
from sklearn.preprocessing import StandardScaler
from sklearn.utils import (
    check_X_y, check_array, check_consistent_length, column_or_1d, safe_mask)
from sklearn.utils.extmath import safe_sparse_dot


def _dia_matrix(X):
    if len(X) == 0:
        return np.array([])
    return sparse.dia_matrix(X)

def _axis0_safe_slice(X, mask, len_mask):
    """
    This mask is safer than safe_mask since it returns an
    empty array, when a sparse matrix is sliced with a boolean mask
    with all False, instead of raising an unhelpful error in older
    versions of SciPy.

    See: https://github.com/scipy/scipy/issues/5361

    Also note that we can avoid doing the dot product by checking if
    the len_mask is not zero in _huber_loss_and_gradient but this
    is not going to be the bottleneck, since the number of outliers
    and non_outliers are typically non-zero and it makes the code
    tougher to follow.
    """
    if len_mask != 0:
        return X[safe_mask(X, mask), :]
    return np.array([])


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

    sample_weight, optional
        Weight assigned to each sample.

    Returns
    -------
    loss: float
        Huber loss.

    gradient: ndarray, shape (n_features + 1,) or (n_features + 2,)
        Returns the derivative of the huber loss with respect to each
        coefficient, intercept and the scale as a vector.
    """
    X_sparse = sparse.issparse(X)
    n_samples, n_features = X.shape
    fit_intercept = n_features + 2 == w.shape[0]
    if fit_intercept:
        intercept = w[-2]
    sigma = w[-1]
    w = w[:X.shape[1]]

    if sample_weight is not None:
        n_samples = np.sum(sample_weight)

    # Calculate the values where |y - X'w -c / sigma| > epsilon
    # The values above this threshold are outliers.
    linear_loss = y - safe_sparse_dot(X, w)
    if fit_intercept:
        linear_loss -= intercept
    abs_linear_loss = np.abs(linear_loss)
    outliers_mask = (abs_linear_loss / sigma) > epsilon

    # Calculate the linear loss due to the outliers.
    # This is equal to (2 * M * |y - X'w -c / sigma| - M**2)*sigma
    outliers = abs_linear_loss[outliers_mask]
    num_outliers = np.count_nonzero(outliers)
    num_non_outliers = X.shape[0] - num_outliers

    if sample_weight is None:
        n_outliers = np.count_nonzero(outliers_mask)
        outlier_loss = (
            2 * epsilon * np.sum(outliers) -
            sigma * n_outliers * epsilon**2)
    else:
        outliers_sw = sample_weight[outliers_mask]
        n_outliers = np.sum(outliers_sw)
        outlier_loss = (
            2 * epsilon * np.sum(outliers_sw * outliers) -
            sigma * n_outliers * epsilon**2)

    # Calculate the quadratic loss due to the non-outliers.-
    # This is equal to |(y - X'w - c)**2 / sigma**2|*sigma
    non_outliers = linear_loss[~outliers_mask]

    if sample_weight is None:
        squared_loss = np.dot(non_outliers.T, non_outliers) / sigma
    else:
        weighted_non_outliers = sample_weight[~outliers_mask] * non_outliers
        weighted_loss = np.dot(weighted_non_outliers.T, non_outliers)
        squared_loss = weighted_loss / sigma

    if fit_intercept:
        grad = np.zeros(n_features + 2)
    else:
        grad = np.zeros(n_features + 1)


    # Gradient due to the squared loss.
    X_non_outliers = -_axis0_safe_slice(X, ~outliers_mask, num_non_outliers)
    if sample_weight is None:
        grad[:n_features] = (
            2 * safe_sparse_dot(non_outliers, X_non_outliers) / sigma)
    else:
        grad[:n_features] = (
            2 * safe_sparse_dot(weighted_non_outliers, X_non_outliers) / sigma)

    # Gradient due to the linear loss.
    outliers_pos = np.logical_and(linear_loss >= 0, outliers_mask)
    outliers_neg = np.logical_and(linear_loss < 0, outliers_mask)
    num_outliers_pos = np.count_nonzero(outliers_pos)
    num_outliers_neg = num_outliers - num_outliers_pos
    X_outliers_pos = _axis0_safe_slice(X, outliers_pos, num_outliers_pos)
    X_outliers_neg = _axis0_safe_slice(X, outliers_neg, num_outliers_neg)

    if sample_weight is None:

        # Summing along any axis of a matrix returns a matrix object.
        outliers_pos_sum = np.squeeze(np.array(X_outliers_pos.sum(axis=0)))
        outliers_neg_sum = np.squeeze(np.array(X_outliers_neg.sum(axis=0)))

        grad[:n_features] -= 2 * epsilon * outliers_pos_sum
        grad[:n_features] += 2 * epsilon * outliers_neg_sum

    else:
        sw_outliers_pos = sample_weight[outliers_pos]
        sw_outliers_neg = sample_weight[outliers_neg]

        if X_sparse:
            weighted_sum = (
                _dia_matrix(sw_outliers_pos) *
                X_outliers_pos).sum(axis=0)
            weighted_sum = np.squeeze(np.array(weighted_sum))
            grad[:n_features] -= 2 * epsilon * weighted_sum

            weighted_sum = (
                _dia_matrix(sw_outliers_neg) *
                X_outliers_neg).sum(axis=0)
            weighted_sum = np.squeeze(np.array(weighted_sum))
            grad[:n_features] += 2 * epsilon * weighted_sum
        else:
            grad[:n_features] -= 2 * epsilon * (
                np.dot(sw_outliers_pos, X_outliers_pos))
            grad[:n_features] += 2 * epsilon * (
                np.dot(sw_outliers_neg, X_outliers_neg))

    # Gradient due to the penalty.
    grad[:n_features] += alpha * 2 * w

    # Gradient due to sigma.
    grad[-1] = n_samples
    grad[-1] -= n_outliers * epsilon**2
    grad[-1] -= squared_loss / sigma

    # Gradient due to the intercept.
    if fit_intercept:
        if sample_weight is None:
            grad[-2] = -2 * np.sum(non_outliers) / sigma
            grad[-2] -= 2 * epsilon * np.sum(outliers_pos)
            grad[-2] += 2 * epsilon * np.sum(outliers_neg)
        else:
            grad[-2] = -2 * np.sum(weighted_non_outliers) / sigma
            grad[-2] -= 2 * epsilon * np.sum(sw_outliers_pos)
            grad[-2] += 2 * epsilon * np.sum(sw_outliers_neg)

    return n_samples * sigma + squared_loss + outlier_loss + alpha * np.dot(w, w), grad


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

    pgtol: float, default 1e-5
        The iteration will stop when max{|proj g_i | i = 1, ..., n} <= pgtol
        where pg_i is the i-th component of the projected gradient.

    Attributes
    ----------
    coef_: array, shape (n_features,)
        Features got by optimizing the huber loss.

    intercept_: float
        Bias factor.

    scale_ : float
        The value by which |y - X'w -c| is scaled down by.

    n_iter_: int
        Number of iterations that fmin_l_bfgs_b has run for, if available.

    References
    ----------
    .. [1] Peter J. Huber, Elvezio M. Ronchetti, Robust Statistics
           Concomitant scale estimates, pg 172
    .. [2] Art B. Owen (2006), A robust hybrid of lasso and ridge regression.
           http://statweb.stanford.edu/~owen/reports/hhu.pdf
    """

    def __init__(self, epsilon=1.35, n_iter=100, alpha=0.0001,
                 warm_start=False, fit_intercept=True, pgtol=1e-05):
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.alpha = alpha
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.pgtol = pgtol

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
        X = check_array(
            X, copy=False, accept_sparse=['csr'], dtype=np.float64)
        y = column_or_1d(y, warn=True)
        y = np.asarray(y, dtype=np.float64)
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
        elif self.warm_start:
            self.coef_ = np.concatenate(
                (self.coef_, [self.intercept_, self.scale_]))

        bounds = np.tile([-np.inf, np.inf], (self.coef_.shape[0], 1))
        bounds[-1][0] = 1.0

        try:
            self.coef_, f, self.dict_ = optimize.fmin_l_bfgs_b(
                _huber_loss_and_gradient, self.coef_, approx_grad=True,
                args=(X, y, self.epsilon, self.alpha, sample_weight),
                maxiter=self.n_iter, pgtol=self.pgtol, bounds=bounds)
        except TypeError:
            self.coef_, f, self.dict_ = optimize.fmin_l_bfgs_b(
                _huber_loss_and_gradient, self.coef_,
                args=(X, y, self.epsilon, self.alpha, sample_weight),
                bounds=bounds)

        self.n_iter_ = self.dict_.get('nit', None)
        self.scale_ = self.coef_[-1]
        if self.fit_intercept:
            self.intercept_ = self.coef_[-2]
        else:
            self.intercept_ = 0.0
        self.coef_ = self.coef_[:X.shape[1]]

        self.loss_ = f
        return self
