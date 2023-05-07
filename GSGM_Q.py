import numpy as np
import time

def GSGM_Q(Q, c, x, verbosity, arls, maxit, eps, fstop, stopcr):
    """
    Implementation of the GS BCGD Method
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
    time_vec = np.zeros(maxit)
    flagls = 0

    _, n = Q.shape

    start_time = time.time()
    time_vec[0] = 0

    # Values for the computation of the o.f.
    Qx = Q @ x
    xQx = x.T @ Qx
    cx = c.T @ x

    g = Qx - c

    fx = 0.5 * xQx - cx

    it = 1

    while flagls == 0:
        # Vectors updating
        if it == 1:
            time_vec[it] = 0
        else:
            time_vec[it] = time.time() - start_time
        fh[it] = fx

        # Gradient evaluation
        v, ind = np.max(np.abs(g)), np.argmax(np.abs(g))
        gi = g[ind]
        Qii = Q[ind, ind]
        d = -gi

        gnr = gi * d
        gnrit[it] = gnr

        # Stopping criteria and test for termination
        if it >= maxniter - 1:
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
            raise ValueError('Unknown stopping criterion')

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
                    alpha *= 0.1

                if alpha <= 1e-20:
                    z = x
                    fz = fx
                    flagls = 1
                    it -= 1
                    break
        else:
            # Exact alpha
            alpha = 1 / Qii
            z[ind] = x[ind] + alpha * d
            fz = fx + alpha * d * gi + 0.5 * (alpha * d) ** 2 * Qii

        x = z
        fx = fz

        # Incremental update for the gradient (good cost in this case!)
        g = g + alpha * d * Q[:, ind].reshape(-1, 1)

        if verbosity > 0:
            print(f'-----------------** {it} **------------------')
            print(f'gnr      = {abs(gnr)}')
            print(f'f(x)     = {fx}')
            print(f'alpha     = {alpha}')

        it += 1

    if it < maxit:
        fh[it:maxit] = fh[it - 1]
        gnrit[it:maxit] = gnrit[it - 1]
        time_vec[it:maxit] = time_vec[it - 1]

    ttot = time.time() - start_time

    return x, it, fx, ttot, fh, time_vec, gnrit