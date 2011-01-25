"""
============================================================
Test with permutations if the features are really correlated
============================================================

In order to test if the quality of a classification score is
significantly due to correlation between variables ie. the
multivariate pattern in the data, it is possible to randomize the
features in a certain way.

See Test 2 in :
Ojala and Garriga. Permutation Tests for Studying Classifier Performance.
The Journal of Machine Learning Research (2010) vol. 11
"""

# Author:  Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD

print __doc__

import numpy as np
import pylab as pl

from scikits.learn import svm
from scikits.learn.cross_val import StratifiedKFold, \
                                    permutation_test_score, \
                                    permute_target, \
                                    permute_features_within_class
from scikits.learn.metrics import zero_one_score


##############################################################################
# Loading a dataset
n_permutations = 10
X = np.random.randn(300, 2)

y_correlated = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
y_uncorrelated = X[:, 0] > 0

y = y_uncorrelated

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))

svm_linear = svm.NuSVC(kernel='linear')
svm_rbf = svm.NuSVC(kernel='rbf')
cv = StratifiedKFold(y, 2)

pl.figure()
from matplotlib import colors
cmap = colors.LinearSegmentedColormap('blue_green_red',
    {'blue' : [(0, 0.7, 0.7), (1, 1, 1)],
     'green' : [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'red' : [(0, 1, 1), (1, 0.7, 0.7)],
    })
fig_ind = 1
for clf in [svm_rbf, svm_linear]:
    for y in [y_correlated, y_uncorrelated]:
        score, _, pvalue_test1 = permutation_test_score(clf, X, y,
                                    zero_one_score, cv=cv,
                                    permute_func=permute_target,
                                    n_permutations=n_permutations, n_jobs=1)

        _, _, pvalue_test2 = permutation_test_score(clf, X, y,
                                    zero_one_score, cv=cv,
                                    permute_func=permute_features_within_class,
                                    n_permutations=n_permutations, n_jobs=1)

        print "Classification score %.2f - " \
              "p-values: %.2f (test 1) & %.2f (test 2)" % (
                                                    score, pvalue_test1,
                                                    pvalue_test2)

        clf.fit(X, y)

        # View scores
        splot = pl.subplot(2, 2, fig_ind)
        fig_ind += 1
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        pl.pcolormesh(xx, yy, Z, cmap=cmap)
        pl.scatter(X[y==0,0], X[y==0,1], c='b')
        pl.scatter(X[y==1,0], X[y==1,1], c='r')
        pl.axis('tight')
        pl.title('p-val. (%.2f, %.2f)' % (pvalue_test1, pvalue_test2))

pl.show()

