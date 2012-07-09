"""
================
Calibration plot
================

"""
print __doc__

# Author: Mathieu Blondel <mathieu@mblondel.org>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD Style.

import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score, calibration_plot
from sklearn.isotonic_regression import isotonic_regression

X, y = make_classification(n_samples=5000, random_state=42)
X_train, y_train = X[:1000], y[:1000]
X_test, y_test = X[1000:], y[1000:]
bins = 8  # for calibration_plot

clf = LogisticRegression(C=len(y_train), intercept_scaling=100.)
clf.fit(X_train, y_train)
prob_pos_lr = clf.predict_proba(X_test)[:, 1]

def isotonic_recalibration(y_test, prob_pos):
    order = np.lexsort((prob_pos, y_test))
    order_inv = np.zeros(len(prob_pos), dtype=np.int)
    order_inv[order] = np.arange(len(prob_pos))
    prob_pos_ir = isotonic_regression(prob_pos[order])
    prob_pos_ir = prob_pos_ir[order_inv]
    return order, prob_pos_ir

order, prob_pos_lr_ir = isotonic_recalibration(y_test, prob_pos_lr)

print "Brier scores: (the smaller the better)"
print "LogisticRegression: %1.3f" % brier_score(y_test, prob_pos_lr)
pt_lr, pp_lr = calibration_plot(y_test, prob_pos_lr, bins=bins)

print "LogisticRegression (IR): %1.3f" % brier_score(y_test, prob_pos_lr_ir)
pt_lr_ir, pp_lr_ir = calibration_plot(y_test, prob_pos_lr_ir, bins=bins)

clf = SGDClassifier(loss="log", seed=0, penalty='l2', n_iter=200)
clf.fit(X_train, y_train)
prob_pos_sgd = clf.predict_proba(X_test)
print "SGD (log): %1.3f" % brier_score(y_test, prob_pos_sgd)
pt_sgd, pp_sgd = calibration_plot(y_test, prob_pos_sgd, bins=bins)

clf = SGDClassifier(loss="modified_huber", seed=0, penalty='l2', n_iter=200)
clf.fit(X_train, y_train)
prob_pos_sgd2 = clf.predict_proba(X_test)
print "SGD (modified_huber): %1.3f" % brier_score(y_test, prob_pos_sgd2)
pt_sgd2, pp_sgd2 = calibration_plot(y_test, prob_pos_sgd2, bins=bins)

###############################################################################
# Plot calibration plots

import pylab as pl

pl.figure()
pl.plot(prob_pos_lr[order], 'x', label='prob(y_)')
pl.plot(prob_pos_lr_ir[order], 'r', linewidth=3, label='prob(y_) (IR)')
pl.plot(y_test[order], 'g', linewidth=3, label='y')
pl.ylim([-0.05, 1.05])
pl.legend(loc="lower right")
pl.show()

pl.figure()
pl.xlabel("Predicted probability")
pl.ylabel("True probability")
pl.plot([0, 1], [0, 1], "b", label="Perfectly calibrated")
pl.plot(pp_lr, pt_lr, "ms-", label="LogisticRegression")
pl.plot(pp_lr_ir, pt_lr_ir, "ks-", label="LogisticRegression (IR)")
pl.plot(pp_sgd, pt_sgd, "rs-", label="SGDClassifier (log)")
pl.plot(pp_sgd2, pt_sgd2, "gs-", label="SGDClassifier (huber)")
pl.legend(loc="lower right")
pl.ylim([-0.05, 1.05])
pl.show()
