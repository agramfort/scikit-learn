# Authors: Fabian Pedregosa <fabian@fseoane.net>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD Style.

import numpy as np


def isotonic_regression(y, w=None, x_min=None, x_max=None):
    """
    Solve the isotonic regression model:

        min Sum w_i (y_i - x_i) ** 2

        subject to x_min = x_1 <= x_2 ... <= x_n = x_max

    where each w_i is strictly positive and each y_i is an arbitrary
    real number.

    Parameters
    ----------
    y : iterable of floating-point values
    w : iterable of floating-point values
        If None w is set to 1 (no weights)

    Returns
    -------
    x : list of floating-point values
    """
    if w is None:
        w = np.ones(len(y), dtype=y.dtype)

    if x_min is not None or x_max is not None:
        y = np.copy(y)
        w = np.copy(w)
        if x_min is not None:
            y[0] = x_min
            w[0] = 1e32
        if x_max is not None:
            y[-1] = x_max
            w[-1] = 1e32

    J = [[i,] for i in range(len(y))]
    cur = 0

    while cur < len(J) - 1:
        av0, av1, av2 = 0, 0, np.inf
        while av0 <= av1 and cur < len(J) - 1:
            idx0 = J[cur]
            idx1 = J[cur + 1]
            av0 = np.dot(w[idx0], y[idx0]) / np.sum(w[idx0])
            av1 = np.dot(w[idx1], y[idx1]) / np.sum(w[idx1])
            cur += 1 if av0 <= av1 else 0

        if cur == len(J) - 1:
            break

        a = J.pop(cur)
        b = J.pop(cur)
        J.insert(cur, a + b)
        while av2 > av0 and cur > 0:
            idx0 = J[cur]
            idx2 = J[cur - 1]
            av0 = np.dot(w[idx0], y[idx0]) / np.sum(w[idx0])
            av2 = np.dot(w[idx2], y[idx2]) / np.sum(w[idx2])
            if av2 >= av0:
                a = J.pop(cur - 1)
                b = J.pop(cur - 1)
                J.insert(cur - 1, a + b)
                cur -= 1

    sol = []
    for idx in J:
        sol += [np.dot(w[idx], y[idx]) / np.sum(w[idx])] * len(idx)
    return np.asarray(sol)

if __name__ == '__main__':

    dat = np.arange(10).astype(np.float)
    dat += 2 * np.random.randn(10)  # add noise

    dat_hat = isotonic_regression(dat, x_min=0, x_max=8)

    import pylab as pl
    pl.close('all')
    pl.plot(dat, 'ro')
    pl.plot(dat_hat, 'b')
    pl.show()
