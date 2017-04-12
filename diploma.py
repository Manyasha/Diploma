from sympy import diff, symbols, Symbol, cos, sin
import numpy as np
from scipy.optimize import minimize
from sympy.utilities.lambdify import lambdify
import sympy as sy

#sy.init_printing()  # LaTeX like pretty printing for IPython

x1, x2, x3, x4 = sy.symbols('x1 x2 x3 x4')

xx = (x1, x2, x3, x4)
f = 100*(x2 - x1**2)**2 + (1 - x1)**2
f_n = lambdify(xx, f, modules='numpy')

# Build Jacobian:
jac_f = [f.diff(x) for x in xx]
jac_fn = [lambdify(xx, jf, modules='numpy') for jf in jac_f]


def f_v(zz):
    """ Helper for receiving vector parameters """
    return f_n(zz[0], zz[1], zz[2], zz[3])


def jac_v(zz):
    """ Jacobian Helper for receiving vector parameters """
    return np.array([jfn(zz[0], zz[1], zz[2], zz[3]) for jfn in jac_fn])


bnds = ((-1, 1), (-1, 1), (-1, 1), (-1, 1))
zz0 = np.array([1, 1, 1, 1])

rslts = minimize(f_v, zz0, method='SLSQP', jac=jac_v, bounds=bnds)
print(rslts)


def d():
    return x1**3 + x2**2;

def f1():
    return 100*(x2 - x1**2)**2 + (1 - x1)**2

def gradient(f, x0):
    print('---------')
    x = np.array([x1, x2, x3, x4])
    n = len(x0)
    grad = np.zeros(n)
    for i in range(n):
        grad[i] = diff(f, x[i]).subs(x0).n()
    return grad

#print(gradient(d(), np.array([2], dtype=float)))
print(gradient(f1(),{x1: -1.2, x2: 1}))

