"""
======================================================================
Calinski-Harabasz score for different number of clusters using K-Means
======================================================================


"""
print __doc__

from math import log
import numpy as np
from scikits.learn.cluster import KMeans
from scikits.learn.metrics import calinski_score
from scikits.learn.datasets.samples_generator import make_blobs

###############################################################################
# Generate sample data
means = np.array([[0, 0], [-1, -1], [1, -1]])
X, _ = make_blobs(n_samples=1500, centers=means, cluster_std=0.2)

###############################################################################
# Compute Calinski-Harabasz scores for different number of clusters

km = KMeans()
scores = []
possible_n_clusters = range(2, 9)
for k in possible_n_clusters:
    labels = km.fit(X, k=k).labels_
    scores.append(log(calinski_score(X, labels)))

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
pl.ylabel('Log(Calinski-Harabasz index)')
pl.show()

pl.figure(2)
pl.clf()
pl.plot(X[:, 0], X[:, 1], '+')
pl.show()
