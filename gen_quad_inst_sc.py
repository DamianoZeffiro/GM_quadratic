import numpy as np

def gen_quad_inst_sc(n, l, ndeg, ncond):
    """
    Instance generator for:
    min 0.5*x'Qx-c'x
    s.t. e'x=1, x>=0

    Input:
    n: dimension of the problem
    l: sparsity of the optimal solution
    ndeg: amount of degeneracy
    ncond: condition number of Q

    Output:
    xstar: optimal solution
    Q: Hessian of the problem
    c: linear term of the objective function
    d: diagonal of Q
    """
    # Generate random vector in (-1, 1)
    y = np.random.rand(n, 1)
    y = -1 + 2 * y

    # Generate matrix Y
    Y = y @ y.T
    Y = np.eye(n) - 2 / np.linalg.norm(y, 2)**2 * Y

    # Generate vector d
    d = np.exp(np.arange(n) / (n - 1) * ncond)

    # Generate Q matrix
    Q = Y @ np.diag(d) @ Y

    # Generate optimal solution xstar
    u = np.random.rand(1, l)
    e = -np.log(u)
    p = np.diag(1 / np.sum(e, axis=1)) @ e

    q = np.random.permutation(n)
    xstar = np.zeros((n, 1))

    xstar[q[:l]] = p.reshape(-1, 1)

    # Generate r vector
    r = np.ones((n, 1))
    mu = np.random.rand(n - l, 1)
    eps = 10**(-mu * ndeg)

    r[q[l:]] = r[q[l:]] + eps

    # Generate c
    c = Q @ xstar - r

    return xstar, Q, c, d
