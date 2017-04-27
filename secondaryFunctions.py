import numpy as np
from sympy import diff, symbols
from sympy.utilities.lambdify import lambdify

#sy.init_printing()  # LaTeX like pretty printing for IPython

x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
x_array = (x1, x2, x3, x4)

def Gradient(f, x):
    gradient_f = [diff(f,x_comp) for x_comp in x_array[0:len(x)]]
    gradient_fn = [lambdify(x_array[0:len(x)], gradient_fn_comp, modules='numpy') for gradient_fn_comp in gradient_f]
    return np.array([gradient_el(*x) for gradient_el in gradient_fn])
