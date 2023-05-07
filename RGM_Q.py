import numpy as np
import time


def RGM_Q(Q, c, x, verbosity, arls, maxit, eps, fstop, stopcr):
    """
    Implementation of the Randomized BCGD Method
    for min f(x) = 0.5 x'Qx - cx

    INPUTS:
    Q: Hessian matrix
    c: linear term
    x: starting point
    verbosity: printing level
    arls: line search (1 Armijo 2 exact)
    maxit: maximum number of iterations
    eps: tolerance
    fstop: target o.f. value
    stopcr: stopping condition
    """

    gamma = 0.0001
    maxniter = maxit
    fh = np.zeros(maxit)
    gnrit = np.zeros(maxit)
    timeVec = np.zeros(maxit)
    flagls = 0

    n = Q.shape[1]

    t_start = time.time()
    timeVec[0] = 0

    # Values for the computation of the o.f.
    Qx = Q @ x
    xQx = x.T @ Qx
    cx = c.T @ x

    fx = 0.5 * xQx - cx

    it = 1

    while flagls == 0:
        # Vectors updating
        if it == 1:
            timeVec[it - 1] = 0
        else:
            timeVec[it - 1] = time.time() - t_start
        fh[it - 1] = fx

        # Gradient evaluation
        while True:
            ind = np.random.randint(n)
            Qix = Q[ind, :] @ x
            ci = c[ind]
            gi = Qix - ci
            Qii = Q[ind, ind]
            d = -gi
            if d != 0.0:
                break

        gnr = gi * d
        gnrit[it - 1] = gnr

        # Stopping criteria and test for termination
        if it >= maxniter:
            break

        if stopcr == 1:
            # Continue if not yet reached target value fstop
            if fx <= fstop:
                break
        elif stopcr == 2:
            # Stopping criterion based on the product of the
            # gradient with the direction
            if abs(n * gnr) <= eps:
                break
        else:
            raise ValueError("Unknown stopping criterion")

        # Line search
        # Set z = x
        z = x.copy()

        if arls == 1:
            # Armijo search
            alpha = 1.0
            ref = gamma * gnr

            while True:
                z[ind] = x[ind] + alpha * d

                # Smart computation of the o.f. at the trial point
                fz = fx + alpha * d * gi + 0.5 * (alpha * d) ** 2 * Qii

                if fz <= fx + alpha * ref:
                    z[ind] = x[ind] + alpha * d
                    break
                else:
                    alpha = alpha * 0.1

                if alpha <= 1e-20:
                    z = x.copy()
                    fz = fx
                    flagls = 1
                    it = it - 1
                    break

        else:
            # Exact alpha
            alpha = 1 / Qii
            z[ind] = x[ind] + alpha * d
            fz = fx
            fz = fx + alpha * d * gi + 0.5 * (alpha * d) ** 2 * Qii

        x = z.copy()
        fx = fz

        if verbosity > 0:
            print('-----------------** {} **------------------'.format(it))
            print('gnr      = {}'.format(abs(gnr)))
            print('f(x)     = {}'.format(fx))
            print('alpha    = {}'.format(alpha))

        it = it + 1

    if it < maxit:
        fh[it:] = fh[it - 1]
        gnrit[it:] = gnrit[it - 1]
        timeVec[it:] = timeVec[it - 1]

    ttot = time.time() - t_start

    return x, it, fx, ttot, fh, timeVec, gnrit
