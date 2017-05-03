import numpy as np
import cmath as cm
from math import sqrt
from sympy import diff, symbols, Symbol
from sympy.solvers import solve
from sympy.utilities.lambdify import lambdify
from scipy.optimize import minimize

#sy.init_printing()  # LaTeX like pretty printing for IPython

x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
betta = Symbol('betta', real=True, positive=True)
x_array = (x1, x2, x3, x4)

def Gradient(f, x):
    gradient_f = [diff(f,x_comp) for x_comp in x_array[0:len(x)]]
    gradient_fn = [lambdify(x_array[0:len(x)], gradient_fn_comp, modules='numpy') for gradient_fn_comp in gradient_f]
    return np.array([gradient_el(*x) for gradient_el in gradient_fn])

def Hessian(f, x):
    hessian_f = [[lambdify(x_array[0:len(x)], diff(f,x_i,x_j), modules='numpy') for x_i in x_array[0:len(x)]] for x_j in x_array[0:len(x)]]
    hessian_fn = [[hessian_f[h_i][h_j](*x) for h_i in range(len(x))] for h_j in range(len(x))]  
    #print([[diff(f,x_i,x_j) for x_i in x_array[0:len(x)]] for x_j in x_array[0:len(x)]])
    return np.matrix(hessian_fn)

def norm(x):
    return sqrt(sum([x_i**2 for x_i in x]))

def BreakCriterion(f, x, eps):
    return norm(Gradient(f, x)) < eps

def gamma(f, x, s):
    numerator = np.matrix(Gradient(f, x))*(Hessian(f, x)*np.matrix(s).T)
    denominator = np.matrix(s)*(Hessian(f, x)*np.matrix(s).T)
    return numerator / denominator

def findStep(f, x_k, s_k):
    point = x_k + betta*s_k
    mod_f = lambdify(x_array[0:len(x_k)], f, modules='numpy')
    betta_array = np.array([el for el in solve(diff(mod_f(*point), betta), rational=None, cubics=False, quartics=False) if el > 0 or el == 0])
    print(betta_array)
    betta_min = 0 if len(betta_array) == 0 else betta_array.min()
    if len(betta_array) == 0:
        print("Betta Array length = 0. Something went wrong")
    print(solve(diff(mod_f(*point), betta), rational=None, cubics=False, quartics=False))
    #f_n = lambdify(betta, mod_f(*point), modules='numpy')
    #betta_min = minimize(f_n, np.array([0]), method='Powell').x
    return betta_min
    
