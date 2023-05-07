import numpy as np
import time

def GM_Q(Q, c, x, verbosity, maxit, eps, fstop, stopcr, lc, arls):
    gamma = 0.0001
    maxniter = maxit
    fh = np.zeros(maxit)
    gnrit = np.zeros(maxit)
    time_vec = np.zeros(maxit)
    flagls = 0

    start_time = time.time()
    time_vec[0] = 0

    # Values for the smart computation of the o.f.
    Qx = Q @ x
    xQx = x.T @ Qx
    cx = c.T @ x

    fx = 0.5 * xQx - cx

    it = 1

    while flagls == 0:
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

        if it >= maxniter:
            break

        if stopcr == 1:
            if fx <= fstop:
                break
        elif stopcr == 2:
            if abs(gnr) <= eps:
                break
        else:
            raise ValueError("Unknown stopping criterion")

        # Line search
        arls = 2

        if arls == 1:
            # Armijo search
            alpha = 3
            ref = gamma * gnr

            while True:
                z = x + alpha * d
                Qz = Q @ z
                zQz = z.T @ Qz
                cz = c.T @ z
                fz = 0.5 * zQz - cz

                if fz <= fx + alpha * ref:
                    z = x + alpha * d
                    break
                else:
                    alpha = alpha * 0.1

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

        x = z
        Qx = Qz
        fx = fz

        if verbosity > 0:
            print(f"-----------------** {it} **------------------")
            print(f"gnr      = {abs(gnr)}")
            print(f"f(x)     = {fx}")
            print(f"alpha     = {alpha}")

        it = it + 1

    if it < maxit:
        fh[it:] = fh[it - 1]
        gnrit[it:] = gnrit[it - 1]
        time_vec[it:] = time_vec[it - 1]

    ttot = time.time() - start_time

    return x, it, fx, ttot, fh, time_vec, gnrit