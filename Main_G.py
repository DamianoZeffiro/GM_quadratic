from gen_quad_inst_sc import gen_quad_inst_sc
from GSGM_Q import GSGM_Q
from GM_Q import GM_Q
from AGM_Q import AGM_Q
import numpy as np
import matplotlib.pyplot as plt
from RGM_Q import RGM_Q

# Optimality tolerance:
eps = 1.0e-4
# Stopping criterion
# 1 : reach of a target value for the obj.func. fk - fstop <= eps
# 2 : nabla f(xk)'dk <= eps
stopcr = 2

# verbosity =0 doesn't display info, verbosity =1 display info
verb = 0

# Generation of the instance
n = 2**11
spc = 0.01
T = int(spc * n)
ndeg = 1
ncond = 5

xstar, Q, c, d = gen_quad_inst_sc(n, T, ndeg, ncond)

lc = max(d)

# starting point
x1 = np.zeros((n, 1))

fstop = 0
maxit = 10000
arls = 3

print('*****************')
print('*  GM STANDARD  *')
print('*****************')

xgm, itergm, fxgm, tottimegm, fhgm, time_vec_gm, gnrgm = GM_Q(Q, c, x1, verb, maxit, eps, fstop, stopcr, lc, arls)
# xgm, itergm, fxgm, tottimegm, fhgm, time_vec_gm, gnrgm = AGM_Q(Q, c, x1, lc, verb, arls, maxit, eps, fstop, stopcr)
# xgm, itergm, fxgm, tottimegm, fhgm, time_vec_gm, gnrgm = GSGM_Q(Q, c, x1, verb, arls, maxit, eps, fstop, stopcr)
# xgm, itergm, fxgm, tottimegm, fhgm, time_vec_gm, gnrgm = RGM_Q(Q, c, x1, verb, arls, maxit, eps, fstop, stopcr)

# Print results
print(f"0.5*xQX - cx  = {fxgm[0, 0]:.3e}")
print(f"Number of iterations = {itergm}")
print(f"||gr||^2 = {gnrgm[maxit-1]}")
print(f"CPU time so far = {tottimegm:.3e}")

# plot figure
fmin = np.min(fhgm)

plt.figure()
plt.semilogy(time_vec_gm, fhgm - fmin, 'r-')
plt.title('Gradient Method  - objective function')
plt.legend(['GM'])
plt.xlim([0, 50])
plt.xlabel('time')
plt.ylim([10**(-5), 10**4])
plt.ylabel('err')
plt.show()

# plot figure
plt.figure()
plt.semilogy(fhgm - fmin, 'r-')
plt.title('Gradient Method  - objective function')
plt.legend(['GM'])
plt.xlim([0, 10000])
plt.xlabel('iter')
plt.ylim([10**(-5), 10**4])
plt.ylabel('err')
plt.show()
