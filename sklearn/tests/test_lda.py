import numpy as np

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_raises

from sklearn import lda

# Data is just 6 separable points in the plane
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]], dtype='f')
y = np.array([1, 1, 1, 2, 2, 2])
y3 = np.array([1, 1, 2, 2, 3, 3])

# Degenerate data with only one feature (still should be separable)
X1 = np.array([[-2, ], [-1, ], [-1, ], [1, ], [1, ], [2, ]], dtype='f')


solvers = ['svd', 'lsqr', 'eigen']
alphas = ['ledoit_wolf', 0, None, 0.43, 'dummy', -22.9]


def test_lda_predict():
    """Test LDA classification.

    This checks that LDA implements fit and predict and returns correct values
    for simple toy data.
    """
    for solver in solvers:
        for alpha in alphas:
            clf = lda.LDA(solver=solver, alpha=alpha)
            try:
                y_pred = clf.fit(X, y).predict(X)
            except ValueError:
                assert_raises(ValueError)
                continue
            assert_array_equal(y_pred, y, 'solver ' + str(solver))

            # Assert that it works with 1D data
            y_pred1 = clf.fit(X1, y).predict(X1)
            assert_array_equal(y_pred1, y, 'solver ' + str(solver))

            # Test probability estimates
            y_proba_pred1 = clf.predict_proba(X1)
            assert_array_equal((y_proba_pred1[:, 1] > 0.5) + 1, y,
                               'solver ' + str(solver))
            y_log_proba_pred1 = clf.predict_log_proba(X1)
            assert_array_almost_equal(np.exp(y_log_proba_pred1), y_proba_pred1,
                                      8, 'solver ' + str(solver))

            # Primarily test for commit 2f34950 -- "reuse" of priors
            y_pred3 = clf.fit(X, y3).predict(X)
            # LDA shouldn't be able to separate those
            assert_true(np.any(y_pred3 != y3), 'solver ' + str(solver))
    # Test unknown solver
    clf = lda.LDA(solver="dummy")
    try:
        y_pred = clf.fit(X, y).predict(X)
    except ValueError:
        assert_raises(ValueError)


def test_lda_transform():
    """Test LDA transform.
    """
    for solver in solvers:
        clf = lda.LDA(solver=solver)
        try:
            X_transformed = clf.fit(X, y).transform(X)
        except NotImplementedError:
            assert_raises(NotImplementedError)
            continue
        assert_equal(X_transformed.shape[1], 1, 'using solver ' + str(solver))


def test_lda_orthogonality():
    # arrange four classes with their means in a kite-shaped pattern
    # the longer distance should be transformed to the first component, and
    # the shorter distance to the second component.
    means = np.array([[0, 0, -1], [0, 2, 0], [0, -2, 0], [0, 0, 5]])

    # We construct perfectly symmetric distributions, so the LDA can estimate
    # precise means.
    scatter = np.array([[0.1, 0, 0], [-0.1, 0, 0], [0, 0.1, 0], [0, -0.1, 0],
                        [0, 0, 0.1], [0, 0, -0.1]])

    X = (means[:, np.newaxis, :] + scatter[np.newaxis, :, :]).reshape((-1, 3))
    y = np.repeat(np.arange(means.shape[0]), scatter.shape[0])

    for solver in solvers:
        # Fit LDA and transform the means
        clf = lda.LDA(solver=solver).fit(X, y)
        try:
            means_transformed = clf.transform(means)
        except NotImplementedError:
            assert_raises(NotImplementedError)
            continue

        d1 = means_transformed[3] - means_transformed[0]
        d2 = means_transformed[2] - means_transformed[1]
        d1 /= np.sqrt(np.sum(d1**2))
        d2 /= np.sqrt(np.sum(d2**2))

        # the transformed within-class covariance should be the identity matrix
        assert_almost_equal(np.cov(clf.transform(scatter).T), np.eye(2))

        # the means of classes 0 and 3 should lie on the first component
        assert_almost_equal(np.abs(np.dot(d1[:2], [1, 0])), 1.0)

        # the means of classes 1 and 2 should lie on the second component
        assert_almost_equal(np.abs(np.dot(d2[:2], [0, 1])), 1.0)


def test_lda_scaling():
    """Test if classification works correctly with differently scaled features.
    """
    n = 100
    # use uniform distribution of features to make sure there is absolutely no
    # overlap between classes.
    x1 = np.random.uniform(-1, 1, (n, 3)) + [-10, 0, 0]
    x2 = np.random.uniform(-1, 1, (n, 3)) + [10, 0, 0]
    x = np.vstack((x1, x2)) * [1, 100, 10000]
    y = [-1] * n + [1] * n

    for solver in solvers[:-1]:  # don't test last unknown solver
        clf = lda.LDA(solver=solver)
        # should be able to separate the data perfectly
        assert_equal(clf.fit(x, y).score(x, y), 1.0,
                     'using covariance: ' + str(solver))
