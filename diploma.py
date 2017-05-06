from sympy import diff, symbols, Symbol, cos, sin
from sympy.solvers import solve
import numpy as np
import math
from conjugateGradientMethods import fourStepsCGM, threeStepsCGM
from scipy.optimize import minimize
from sympy.utilities.lambdify import lambdify
import sympy as sy
import secondaryFunctions as sf
#import matplotlib.pyplot as plt
#from numdifftools import Gradient
#import numdifftools.nd_algopy as nda

#sy.init_printing()  # LaTeX like pretty printing for IPython

x1, x2, x3, x4 = sy.symbols('x1 x2 x3 x4')
t_k = sy.symbols('t_k')

xx = (x1, x2)
x_array = (x1, x2, x3, x4)


f = 100*(x2 - x1**2)**2 + (1 - x1)**2
f1 = (x1 + 10*x2)**2 + 5*(x3 - x4)**2 + (x2 - 2*x3)**4 + 10*(x1 - x4)**4
f2 = (x1*x2)**2 * (1 - x1**2) * (1 - x1 - x2*(1 - x1)**5)**2
f3 = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

print(type(f))

f_n = lambdify(xx, f, modules='numpy')
#print(f)
#print(np.array(np.matrix([1,2])).flatten())
#print('----')
#print(sf.findStep(f, np.array([-1.2, 1]), np.array([-215.6, -88])))
#print('----')


print('----')
fourStepsCGM(f, [-1.2, 1], 0.01)
print('----')
threeStepsCGM(f, [-1.2, 1], 0.01)
print('----')
fourStepsCGM(f1, [3,-1,0,1], 0.01)
print('----')
threeStepsCGM(f1, [3,-1,0,1], 0.01)
print('----')
fourStepsCGM(f1, [1,1,1,1], 0.01)
print('----')
threeStepsCGM(f1, [1,1,1,1], 0.01)
print('----')
fourStepsCGM(f3, [1,1], 0.01)
print('----')
threeStepsCGM(f3, [1,1], 0.01)
print('----')





k_test = np.array([5, 1]) + t_k*np.array([2, 2])

print(k_test)
print(f_n(*k_test))
print(solve(f))

#dfun = nd.Gradient(f_n)
#print(dfun([1,2,3]))

# Build Jacobian:
jac_f = [f.diff(x) for x in xx]
jac_fn = [lambdify(xx, jf, modules='numpy') for jf in jac_f]

def f_v(zz):
    """ Helper for receiving vector parameters """
    return f_n(zz[0], zz[1])

def jac_v(zz):
    """ Jacobian Helper for receiving vector parameters """
    return np.array([jfn(zz[0], zz[1]) for jfn in jac_fn])

bnds = ((-1, 1), (-1, 1))
zz0 = np.array([2, 2])

rslts = minimize(f_v, zz0, method='SLSQP', jac=jac_v, bounds=bnds)
res0 = minimize(f_v, zz0, method='Powell')
res1 = minimize(f_v, zz0)

print(rslts.x, rslts.nit)
print(res0.x, res0.nit)
print(res1.x, res1.nit)

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

#print(sf.Hessian(f,[0,0]))
#print(sf.Hessian(f,[0,0])*sf.Gradient(f,[-1.2,1]).T)
x = 5
s_k = {
    0: lambda x: x * 5,
    1: lambda x: x + 7,
    2: lambda x: x + 7
    }.get(0,x - 2)(2)
print(s_k)

print(sf.Gradient(f, [0,0]))
print(sf.norm(sf.Gradient(f, [0,0])))
print(sf.BreakCriterion(f, [1,1], 0.01))
print(sf.gamma(f, [0,0], [1,1]))
print(np.matrix([1,2]))
print(np.matrix([2,4]).T)
print(np.matrix([1,2])*np.matrix([2,4]).T)

#print(gradient(d(), np.array([2], dtype=float)))
print(gradient(f1(),{x1: -1.2, x2: 1}))

