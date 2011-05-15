"""
======================================================================
Calinsky-Harabasz score for different number of clusters using K-Means
======================================================================


"""
print __doc__

from math import log
import numpy as np
from scikits.learn.cluster import KMeans
from scikits.learn.metrics import calinsky_score

###############################################################################
# Generate sample data
np.random.seed(0)

n_points_per_cluster = 500
n_clusters = 3
n_points = n_points_per_cluster * n_clusters
means = np.array([[0, 0], [-1, -1], [1, -1]])
std = .2

X = np.empty((0, 2))
for i in range(n_clusters):
    X = np.r_[X, means[i] + std * np.random.randn(n_points_per_cluster, 2)]

###############################################################################
# Compute Calinsky-Harabasz scores for different number of clusters

km = KMeans()
scores = []
possible_n_clusters = range(2, 9)
for k in possible_n_clusters:
    labels = km.fit(X, k=k).labels_
    scores.append(log(calinsky_score(X, labels)))

best_n_clusters = possible_n_clusters[np.argmax(scores)]

###############################################################################
# Plot result
import pylab as pl

pl.close('all')
pl.figure(1)
pl.clf()
pl.plot(possible_n_clusters, scores)
pl.title('Estimated number of clusters: %d' % best_n_clusters)
pl.xlabel('Number of clusters')
pl.ylabel('Log(Calinsky-Harabasz index)')
pl.show()

pl.figure(2)
pl.clf()
pl.plot(X[:, 0], X[:, 1], '+')
pl.show()
