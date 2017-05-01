import numpy as np
import secondaryFunctions as sf
from sympy import diff, symbols
from sympy.utilities.lambdify import lambdify
#from mpmath import matrix

#sy.init_printing()  # LaTeX like pretty printing for IPython

x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
x_array = (x1, x2, x3, x4)

def 4stepsCGM(f, x0, eps):
    k = 0
    if sf.BreakCriterion(f, x0, eps):
        print('Stop itaretion')
        break
    
    s_0 = -sf.Gradient(f, x0)
    s_1 = -sf.Gradient(f, x_k) + gamma(f, x_k, s_0)*s_0
    s_2 = -sf.Gradient(f, x_k) + gamma(f, x_k, s_k)*s_0 + gamma(f, x_k, s_k)*s_0
