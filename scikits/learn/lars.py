
import numpy as np
from scipy import linalg
from scikits.learn.glm import LinearModel


# Notes: np.ma.dot copies the masked array before doing the dot
# product. Maybe we should implement in C our own masked_dot that does
# not make unnecessary copies.

# all linalg.solve solve a triangular system, so this could be heavily
# optimized by binding (in scipy ?) trsv or trsm

def lars(X, y, max_iter=None, alpha_min=0, method="lar", precompute=True):
    """
    lar -> m <= x.shape[1]
    lasso -> m can be > x.shape[1]

    precompute : empty for now

    TODO: detect stationary points.
    Lasso variant
    store full path
    """

    n_samples, n_features = X.shape

    if max_iter is None:
        max_iter = min(n_samples, n_features) - 1

    max_pred = max_iter # OK for now

    mu       = np.zeros (X.shape[0])
    beta     = np.zeros ((max_iter + 1, X.shape[1]))
    alphas   = np.zeros (max_iter + 1)
    n_iter, n_pred = 0, 0
    active   = list()
    # holds the sign of covariance
    sign_active = np.empty (max_pred, dtype=np.int8)
    drop = False

    # will hold the cholesky factorization
    # only lower part is referenced. We do not create it as
    # empty array because chol_solve calls chkfinite on the
    # whole array, which can cause problems.
    L = np.zeros ((max_pred, max_pred), dtype=np.float64)

    Xt  = X.T
    Xna = Xt.view(np.ma.MaskedArray) # variables not in the active set
                                     # should have a better name

    while 1:

        # Calculate covariance matrix and get maximum
        res = y - np.dot (X, beta[n_iter]) # there are better ways
        Cov = np.ma.dot (Xna, res)

        # print np.ma.dot(Xt[active], res)

        imax    = np.ma.argmax (np.ma.abs(Cov), fill_value=0.) #rename
        Cov_max =  (Cov [imax]) # rename C_max
        Cov[imax] = np.ma.masked

        alpha = np.abs(Cov_max) #sum (np.abs(beta[n_iter]))
        alphas [n_iter] = alpha #sum (np.abs(beta[n_iter]))
        # print n_iter, imax, Cov_max, np.ma.abs(Cov.data)

        if (n_iter >= max_iter or n_pred >= max_pred or alpha < alpha_min):
            break

        if not drop:

            # Update the Cholesky factorization of (Xa * Xa') #
            #                                                 #
            #          ( L   0 )                              #
            #   L  ->  (       )  , where L * w = b           #
            #          ( w   z )    z = 1 - ||w||             #
            #                                                 #
            #   where u is the last added to the active set   #

            n_pred += 1
            active.append(imax)
            Xna[imax] = np.ma.masked
            sign_active [n_pred-1] = np.sign (Cov_max)

            X_max = Xt[imax]

            c = np.dot (X_max, X_max)
            L [n_pred-1, n_pred-1] = c

            if n_pred > 1:
                b = np.dot (X_max, Xa.T)

                # please refactor me, using linalg.solve is overkill
                L [n_pred-1, :n_pred-1] = linalg.solve (L[:n_pred-1, :n_pred-1], b)
                v = np.dot(L [n_pred-1, :n_pred-1], L [n_pred - 1, :n_pred -1])
                L [n_pred-1,  n_pred-1] = np.sqrt (c - v)
        else:
            drop = False

        Xa = Xt[active] # also Xna[~Xna.mask]

        # Now we go into the normal equations dance.
        # (Golub & Van Loan, 1996)

        b = np.copysign (Cov_max.repeat(n_pred), sign_active[:n_pred])
        b = linalg.cho_solve ((L[:n_pred, :n_pred], True),  b)

        C = A = np.abs(Cov_max)
        u = np.dot (Xa.T, b)
        a = np.ma.dot (Xna, u)

        # equation 2.13, there's probably a simpler way
        g1 = (C - Cov) / (A - a)
        g2 = (C + Cov) / (A + a)

        g = np.ma.concatenate((g1, g2))
        g = g[g >= 0.]
        gamma_ = np.ma.min (g)

        if method == 'lasso':

            z = - beta[n_iter, active] / b
            z[z <= 0.] = np.inf

            idx = np.argmin(z)

            if z[idx] < gamma_:
                gamma_ = z[idx]
                drop = True

        n_iter += 1
        beta[n_iter, active] = beta[n_iter - 1, active] + gamma_ * b

        if drop:
            n_pred -= 1
            drop_idx = active.pop (idx)
            print 'dropped ', idx, ' at ', n_iter, ' iteration'
            Xa = Xt[active] # duplicate
            L[:n_pred, :n_pred] = linalg.cholesky(np.dot(Xa, Xa.T), lower=True)
            sign_active = np.delete(sign_active, idx) # do an append to maintain size
            Xna.mask[drop_idx] = False
            # should be done using cholesky deletes

    if alpha < alpha_min: # interpolate
        # interpolation factor 0 <= ss < 1
        ss = (alphas[n_iter-1] - alpha_min) / (alphas[n_iter-1] - alphas[n_iter])
        beta[n_iter] = beta[n_iter-1] + ss*(beta[n_iter] - beta[n_iter-1]);
        alphas[n_iter] = alpha_min
        alphas = alphas[:n_iter+1]
        beta = beta[:n_iter+1]

    return alphas, active, beta.T


class LARS (LinearModel):

    def __init__(self, n_features, normalize=True):
        self.n_features = n_features
        self.normalize = normalize
        self.coef_ = None

    def fit (self, X, Y):
                # will only normalize non-zero columns

        if self.normalize:
            self._xmean = X.mean(0)
            self._ymean = Y.mean(0)
            X = X - self._xmean
            Y = Y - self._ymean
            self._norms = np.apply_along_axis (np.linalg.norm, 0, X)
            nonzeros = np.flatnonzero(self._norms)
            X[:, nonzeros] /= self._norms[nonzeros]

        method = 'lar'
        alphas_, active, coef_path_ = lars (X, Y,
                                max_iter=self.n_features, method=method)
        print alphas_
        self.coef_ = coef_path_[:,-1]
        return self


class LassoLARS (LinearModel):

    def __init__(self, alpha, normalize=True):
        self.alpha = alpha
        self.normalize = normalize
        self.coef_ = None

    def fit (self, X, Y):
                # will only normalize non-zero columns

        n_samples = X.shape[0]
        alpha = self.alpha * n_samples # scale alpha with number of samples

        if self.normalize:
            self._xmean = X.mean(0)
            self._ymean = Y.mean(0)
            X = X - self._xmean
            Y = Y - self._ymean
            self._norms = np.apply_along_axis (np.linalg.norm, 0, X)
            nonzeros = np.flatnonzero(self._norms)
            X[:, nonzeros] /= self._norms[nonzeros]

        method = 'lasso'
        alphas_, active, coef_path_ = lars (X, Y,
                                            alpha_min=alpha, method=method)
        self.coef_ = coef_path_[:,-1]
        return self

if __name__ == '__main__':

    from scikits.learn.datasets import load_diabetes
    from scikits.learn.glm import Lasso

    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # lar = LARS(n_features=7, normalize=False)
    # lar.fit(X, y)
    # print lar.coef_
    # print np.max(np.abs(np.dot(X.T, y - np.dot(X, lar.coef_))))

    lasso_lars = LassoLARS(alpha=0.1)
    lasso_lars.fit(X, y)
    print np.max(np.abs(np.dot(X.T, y - np.dot(X, lasso_lars.coef_))))
    print lasso_lars.coef_

    # make sure results are the same than with Lasso Coordinate descent
    lasso = Lasso(alpha=0.1)
    lasso.fit(X, y, maxit=3000, tol=1e-10)
    print np.max(np.abs(np.dot(X.T, y - np.dot(X, lasso.coef_))))
    print lasso.coef_

    print "Error : %s " % np.linalg.norm(lasso_lars.coef_ - lasso.coef_)

    # plot full lasso path
    alphas_, active, coef_path_ = lars (X, y, 8, method="lasso")

    import pylab as pl
    pl.close('all')
    pl.plot(-np.log(alphas_), coef_path_.T)
    pl.show()

