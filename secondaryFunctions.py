import numpy as np
import cmath as cm
from math import sqrt
from sympy import diff, symbols, Symbol
from sympy.solvers import solve
from sympy.utilities.lambdify import lambdify
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

#sy.init_printing()  # LaTeX like pretty printing for IPython

x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
beta = Symbol('beta', real=True, positive=True)
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
    numerator = np.matrix(Gradient(f, x))@(Hessian(f, x)*np.matrix(s).T)
    denominator = np.matrix(s)@(Hessian(f, x)*np.matrix(s).T)
    return numerator / denominator

def findStep(f, x_k, s_k):
    point = x_k + beta*s_k
    mod_f = lambdify(x_array[0:len(x_k)], f, modules='numpy')
    mod_fn = lambdify(beta, mod_f(*point), modules='numpy')       
    beta_min = minimize_scalar(mod_fn).x
    return 0 if beta_min < 0 else beta_min

def printInfo(f, x0, ExpectedRes, ActualResFourCGM, ActualResThreeCGM):
    test_f = "Test function: " + f
    test_point = "Initial point: " + x0
    exRes = "Expected Result: x* = " + ExpectedRes.x_star + " f* = " + ExpectedRes.f_star
    acFourRes = "Actual Result 4 steps CGM: x* = " + ActualResFourCGM.x_star + " f* = " + ActualResFourCGM.f_star + " for k = %d" %(ActualResFourCGM.k) + " steps"
    acThreeRes = "Actual Result 3 steps CGM: x* = " + ActualResThreeCGM.x_star + " f* = " + ActualResThreeCGM.f_star + " for k = %d" %(ActualResThreeCGM.k) + " steps"
    breakLine = "/n"
    print(test_f + breakLine + test_point + breakLine + exRes + breakLine + acRes)
