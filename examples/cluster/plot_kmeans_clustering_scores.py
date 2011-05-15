"""
================================================================
Clustering scores for different number of clusters using K-Means
================================================================


"""
print __doc__

from math import log
import numpy as np
from scikits.learn.cluster import KMeans
from scikits.learn.metrics import calinski_score, davies_bouldin_score
from scikits.learn.datasets.samples_generator import make_blobs

###############################################################################
# Generate sample data
means = np.array([[0, 0], [-1, -1], [1, -1]])
X, _ = make_blobs(n_samples=1500, centers=means, cluster_std=0.2)

###############################################################################
# Compute Calinski-Harabasz scores for different number of clusters

km = KMeans()
ch_scores = []
db_scores = []
possible_n_clusters = range(2, 9)
for k in possible_n_clusters:
    labels = km.fit(X, k=k).labels_
    ch_scores.append(log(calinski_score(X, labels)))
    db_scores.append(davies_bouldin_score(X, labels))

###############################################################################
# Plot result
import pylab as pl

pl.close('all')
pl.figure(1)
pl.clf()
pl.subplot(2, 1, 1)
pl.plot(possible_n_clusters, ch_scores)
pl.title('Estimated number of clusters: %d' %
                                possible_n_clusters[np.argmax(ch_scores)])
pl.xlabel('Number of clusters')
pl.ylabel('Log(Calinski-Harabasz index)')

pl.subplot(2, 1, 2)
pl.plot(possible_n_clusters, db_scores)
pl.title('Estimated number of clusters: %d' %
                                possible_n_clusters[np.argmin(db_scores)])
pl.xlabel('Number of clusters')
pl.ylabel('Davies-Bouldin index')
pl.subplots_adjust(0.09, 0.09, 0.94, 0.94, 0.26, 0.36)
pl.show()