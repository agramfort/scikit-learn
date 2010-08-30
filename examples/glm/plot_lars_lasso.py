#!/usr/bin/env python
"""
======================
Least Angle Regression
======================

"""

# Author: Fabian Pedregosa <fabian.pedregosa@inria.fr>
# License: BSD Style.

from datetime import datetime
import itertools
import numpy as np
import pylab as pl

from scikits.learn import lars, glm
from scikits.learn import datasets

diabetes = datasets.load_diabetes()
X = diabetes.data
Y = diabetes.target

np.random.seed(0)
n, m = 30, 50
X = np.random.randn(n, m)
Y = np.random.randn(n)

################################################################################
# Demo path functions
################################################################################

print "Computing regularization path using the LARS ..."
start = datetime.now()
clf = lars.LassoLARS().fit (X, Y, max_features=28, normalize=True)
print "This took ", datetime.now() - start

alphas = -np.log10(clf.alphas_)

# # Display results
color_iter = itertools.cycle (['r', 'g', 'b', 'c'])

for coef_, color in zip(clf.coef_path_, color_iter):
    pl.plot(alphas, coef_.T, color)

ymin, ymax = pl.ylim()
pl.vlines(alphas, ymin, ymax, linestyle='dashed')
pl.xlabel('-Log(lambda)')
pl.ylabel('weights')
pl.title('Least Angle Regression (LAR) Paths')
pl.axis('tight')
pl.show()

