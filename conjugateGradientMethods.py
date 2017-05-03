import numpy as np
import secondaryFunctions as sf
from sympy import diff, symbols
from sympy.utilities.lambdify import lambdify
#from mpmath import matrix

#sy.init_printing()  # LaTeX like pretty printing for IPython

x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
x_array = (x1, x2, x3, x4)

def fourStepsCGM(f, x0, eps):
    k = 0
    x_k = x0
    s_last = {
        0: 0,
        1: 0,
        2: 0
    }
    s = {
        0: lambda x_k: np.matrix(-sf.Gradient(f, x_k)),
        1: lambda x_k: -sf.Gradient(f, x_k) + sf.gamma(f, x_k, s_last[0])*s_last[0],
        2: lambda x_k: -sf.Gradient(f, x_k) + sf.gamma(f, x_k, s_last[1])*s_last[1] + sf.gamma(f, x_k, s_last[0])*s_last[0],
        3: lambda x_k: -sf.Gradient(f, x_k) + sf.gamma(f, x_k, s_last[2])*s_last[2] + sf.gamma(f, x_k, s_last[1])*s_last[1] + sf.gamma(f, x_k, s_last[0])*s_last[0]
    }

    while not sf.BreakCriterion(f, x0, eps):
        s_k = np.array(s.get(k, s.get(3))(x_k)).flatten()
        if k < 3:
            s_last[k] = s_k
        else:
            s_last[0] = s_last[1]
            s_last[1] = s_last[2]
            s_last[2] = s_k
        
        betta_k = sf.findStep(f, x_k, s_k) 
        x_k1 = x_k + betta_k*s_k
        x_k = x_k1
        k = k + 1
    print(x_k)
    
