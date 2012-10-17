"""
======================================================
Run unsupervised model selection using Factor Analysis
======================================================

"""
print __doc__

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# Licence: BSD

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import FactorAnalysis
from sklearn.grid_search import GridSearchCV

n_samples, n_features, n_components = 1000, 5, 3

# Some random settings for the generative model
rng = np.random.RandomState(42)
W = rng.randn(n_components, n_features)
# latent variable of dim 3, 20 of it
h = rng.randn(n_samples, n_components)
# using gamma to model different noise variance
# per component
noise = 0.5 * rng.gamma(1, size=n_features) * rng.randn(n_samples, n_features)

# generate observations
# wlog, mean is 0
X = np.dot(h, W) + noise

scores = list()
all_n_components = range(1, 8)

for n_components in all_n_components:
    fa = FactorAnalysis(n_components=n_components, tol=1e-3)
    scores.append(np.mean(cross_val_score(fa, X, cv=2)))

print "Scores : %s" % scores
print "Estimated number of components: %d" % all_n_components[np.argmax(scores)]

import pylab as pl
pl.plot(all_n_components, scores)
pl.xlabel('# of components')
pl.ylabel('score')
pl.show()

###############################################################################
# Or using grid search

parameters = dict(n_components=all_n_components)
gs = GridSearchCV(fa, parameters)
gs.fit(X)
print "Grid search n_components : %d", gs.best_estimator_.n_components
