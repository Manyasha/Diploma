import numpy as np
import cmath as cm
from math import sqrt, pow, log10, isinf, fabs
from sympy import diff, symbols, Symbol, init_printing, latex
from sympy.solvers import solve
from sympy.utilities.lambdify import lambdify
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from decimal import Decimal, getcontext
from matplotlib import mlab
import matplotlib.pyplot as plt

init_printing()  # LaTeX like pretty printing for IPython
getcontext().prec = 15

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
    return np.matrix(hessian_fn)

def norm(x):
    try:
        norm = sqrt(sum([pow(x_i,2) for x_i in x]))
    except OverflowError:
        norm = float('inf')
    return norm

def BreakCriterion(f, x_k, x_k_minus_1, eps):
    first = f_at_point(f, x_k_minus_1) - f_at_point(f, x_k) < eps*(1 + fabs(f_at_point(f, x_k)))
    second = norm(np.array(x_k_minus_1) - np.array(x_k)) < sqrt(eps)*(1 + norm(x_k))
    third = norm(Gradient(f, x_k)) <= pow(eps, 1/3)*(1 + fabs(f_at_point(f, x_k)))
    return first and second and third

def gamma(f, x, s):
    numerator = np.matrix(Gradient(f, x))*(Hessian(f, x)*np.matrix(s).T)
    denominator = np.matrix(s)*(Hessian(f, x)*np.matrix(s).T)
    return 0 if denominator == 0 else numerator / denominator

def gamma_non_kvad(f, x_k, x_k_minus_first, x_k_minus_second):
    grad = np.matrix(Gradient(f, x_k_minus_first)) - np.matrix(Gradient(f, x_k_minus_second))
    
    numerator = np.matrix(Gradient(f, x_k))*grad.T
    denominator = pow(norm(Gradient(f, x_k_minus_second)),2)            
    return 0 if denominator == 0 else numerator / denominator

def findStep(f, x_k, s_k):
    point = x_k + beta*s_k
    mod_f = lambdify(x_array[0:len(x_k)], f, modules='numpy')
    mod_fn = lambdify(beta, mod_f(*point), modules='numpy')
    beta_min = minimize_scalar(mod_fn).x
    if beta_min < 0:
        print(beta_min)
    return beta_min

def f_at_point(f, point, isRound = False):
    try:
        f_point = lambdify(x_array[0:len(point)], f, modules='numpy')(*point)
    except OverflowError:
        return float('inf')
    return round(f_point, 5) if isRound == True else f_point

def printInfo(f, x0, eps, ExpectedRes, ActualResFourCGM, ActualResThreeCGM):
    test_f = "Test function: " + str(f)
    test_point = "Initial point: " + str(x0)
    accuracy = "Accuracy of calculations: " + str(eps)
    exRes = "Expected Result: x* = " + str(ExpectedRes['x_star']) + " f* = " + str(ExpectedRes['f_star'])
    acFourRes = "" if len(ActualResFourCGM) == 0 else "Actual Result 4 steps CGM: x* = " + str(ActualResFourCGM['x_star']) + " f* = " + str(ActualResFourCGM['f_star']) + " for k = %d" %(ActualResFourCGM['k']) + " steps"
    acThreeRes = "" if len(ActualResThreeCGM) == 0 else "Actual Result 3 steps CGM: x* = " + str(ActualResThreeCGM['x_star']) + " f* = " + str(ActualResThreeCGM['f_star']) + " for k = %d" %(ActualResThreeCGM['k']) + " steps"
    breakLine = "\n"
    f_at_init_point = "Function at initial point: " + str(f_at_point(f, x0, True))
    print(latex(f))
    print(test_f + breakLine + test_point + breakLine + f_at_init_point + breakLine + accuracy + breakLine + exRes + breakLine + acFourRes + breakLine + acThreeRes + breakLine)

def showPlot(f, fourStepsRes, threeStepsRes):
    f_x_star_fourSteps = f_at_point(f, fourStepsRes['x_star'])
    f_x_star_threeSteps = f_at_point(f, threeStepsRes['x_star'])
    
    fourSteps_f_points = [f_at_point(f, x_i) for x_i in fourStepsRes['x_points']]
    threeSteps_f_points = [f_at_point(f, x_i) for x_i in threeStepsRes['x_points']]
    log_fourSteps = [Decimal(f_point_k - f_x_star_fourSteps).log10() for f_point_k in fourSteps_f_points]
    log_threeSteps = [Decimal(f_point_k - f_x_star_threeSteps).log10() for f_point_k in threeSteps_f_points]

    plt.plot(range(fourStepsRes['k'] + 1), log_fourSteps)
    plt.plot(range(threeStepsRes['k'] + 1), log_threeSteps, 'r--')

    plt.ylabel('lg [f(x_i) - f(x*)]')
    plt.xlabel('i')
    
    plt.show()
