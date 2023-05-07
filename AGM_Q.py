import time
import numpy as np

def AGM_Q(Q, c, x, lc, verbosity, arls, maxit, eps, fstop, stopcr):
    """
    Implementation of the Accelerated Gradient Method
    for min f(x) = 0.5 x'Qx - cx

    INPUTS:
    Q: Hessian matrix
    c: linear term
    x: starting point
    lc: Lipschitz constant of the gradient (not needed if exact/Armijo ls used)
    verbosity: printing level
    arls: line search (1 Armijo, 2 exact, 3 fixed)
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
    lambda_ = 1.0

    start_time = time.time()
    time_vec[0] = 0

    # Values for the smart computation of the o.f.
    Qx = Q @ x
    xQx = x.T @ Qx
    cx = c.T @ x

    fx = 0.5 * xQx - cx

    it = 1

    while flagls == 0:
        # Vectors updating
        if it == 1:
            time_vec[it - 1] = 0
        else:
            time_vec[it - 1] = time.time() - start_time
        fh[it - 1] = fx

        # Gradient evaluation
        g = Qx - c
        d = -g

        gnr = g.T @ d
        gnrit[it - 1] = -gnr

        # Stopping criteria and test for termination
        if it >= maxniter:
            break

        if stopcr == 1:
            # Continue if not yet reached target value fstop
            if fx <= fstop:
                break
        elif stopcr == 2:
            # Stopping criterion based on the product of the gradient with the direction
            if abs(gnr) <= eps:
                break
        else:
            raise ValueError("Unknown stopping criterion")

        # Line search
        if arls == 1:
            # Armijo search
            alpha = 1
            ref = gamma * gnr

            while True:
                z = x + alpha * d
                # Computation of the o.f. at the trial point
                Qz = Q @ z
                zQz = z.T @ Qz
                cz = c.T @ z

                fz = 0.5 * zQz - cz

                if fz <= fx + alpha * ref:
                    z = x + alpha * d
                    break
                else:
                    alpha *= 0.1

                if alpha <= 1e-20:
                    z = x
                    fz = fx
                    flagls = 1
                    it = it - 1
                    break
        elif arls == 2:
            # Exact alpha
            alpha = -gnr / (d.T @ Q @ d)
            z = x + alpha * d
            Qz = Q @ z
            zQz = z.T @ Qz
            cz = c.T @ z
            fz = 0.5 * zQz - cz
        else:
            # Fixed alpha
            alpha = 1 / lc
            z = x + alpha * d
            Qz = Q @ z
            zQz = z.T @ Qz
            cz = c.T @ z
            fz = 0.5 * zQz - cz

        xold = x
        x = z
        Qx = Qz
        fx = fz

        # Acceleration step
        lambda1 = (1 + np.sqrt(1 + 4 * lambda_ ** 2)) / 2
        sta = (lambda_ - 1) / lambda1
        x = x + sta * (x - xold)
        lambda_ = lambda1

        if verbosity > 0:
            print("-----------------**", it, "**------------------")
            print("gnr      =", abs(gnr))
            print("f(x)     =", fx)
            print("alpha     =", alpha)

        it = it + 1

    if it < maxit:
        fh[it:] = fh[it - 1]
        gnrit[it:] = gnrit[it - 1]
        time_vec[it:] = time_vec[it - 1]

    ttot = time.time() - start_time

    return x, it, fx, ttot, fh, time_vec, gnrit