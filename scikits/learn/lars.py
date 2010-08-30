
import numpy as np
from scipy import linalg


# Notes: np.ma.dot copies the masked array before doing the dot
# product. Maybe we should implement in C our own masked_dot that does
# not make unnecessary copies.

# all linalg.solve solve a triangular system, so this could be heavily
# optimized by binding (in scipy ?) trsv or trsm

def lars(X, y, max_iter, method="lar", precompute=True):
    """
    lar -> m <= x.shape[1]
    lasso -> m can be > x.shape[1]

    precompute : empty for now

    TODO: detect stationary points.
    Lasso variant
    store full path
    """

    max_pred = max_iter # OK for now

    mu       = np.zeros (X.shape[0])
    beta     = np.zeros ((max_iter + 1, X.shape[1]))
    alphas   = np.empty (max_iter + 1)
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

        alphas [n_iter] = np.abs(Cov_max) #sum (np.abs(beta[n_iter]))
        # print n_iter, imax, Cov_max, np.ma.abs(Cov.data)

        if (n_iter >= max_iter or n_pred >= max_pred):
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

    return alphas, active, beta.T


class LassoLARS (object):

    def fit (self, X, Y, max_features, method='lasso', normalize=True):
                # will only normalize non-zero columns

        if normalize:
            self._xmean = X.mean(0)
            self._ymean = Y.mean(0)
            X = X - self._xmean
            Y = Y - self._ymean
            self._norms = np.apply_along_axis (np.linalg.norm, 0, X)
            nonzeros = np.flatnonzero(self._norms)
            X[:, nonzeros] /= self._norms[nonzeros]

        self.alphas_, self.active, self.coef_path_ = lars (X, Y, max_features, method=method)
        return self




