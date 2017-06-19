import numpy as np
import secondaryFunctions as sf
from sympy import diff, symbols
from sympy.utilities.lambdify import lambdify

#sy.init_printing()  # LaTeX like pretty printing for IPython

def fourStepsCGM(f, x0, eps):
    k = 0
    x_k = x0
    x_k_minus_1 = np.zeros(len(x_k))
    s_last = {
        0: 0,
        1: 0,
        2: 0
    }
    s = {
        0: lambda x_k: np.matrix(-sf.Gradient(f, x_k)),
        1: lambda x_k: -sf.Gradient(f, x_k) + 
						sf.gamma(f, x_k, s_last[0])*s_last[0],
        2: lambda x_k: -sf.Gradient(f, x_k) + 
						sf.gamma(f, x_k, s_last[1])*s_last[1] +
						sf.gamma(f, x_k, s_last[0])*s_last[0],
        3: lambda x_k: -sf.Gradient(f, x_k) + 
						sf.gamma(f, x_k, s_last[2])*s_last[2] + 
						sf.gamma(f, x_k, s_last[1])*s_last[1] + 
						sf.gamma(f, x_k, s_last[0])*s_last[0]
    }
    x_points = [x_k];

    while not sf.BreakCriterion(f, x_k, x_k_minus_1, eps):
        s_k = np.array(s.get(k, s.get(3))(x_k)).flatten()        
        if k < 3:
            s_last[k] = s_k
        else:
            s_last[0] = s_last[1]
            s_last[1] = s_last[2]
            s_last[2] = s_k
        
        beta_k = sf.findStep(f, x_k, s_k)
        x_k1 = x_k + beta_k*s_k
        x_k_minus_1 = x_k
        #print(sf.f_at_point(f, x_k, True))
        x_k = x_k1
        k = k + 1
        x_points.append(x_k)
        
    f_star = sf.f_at_point(f, x_k, True)
    return {'x_star':x_k,'f_star':f_star,'k':k,'x_points':x_points}

def nonQvadFourStepsCGM(f, x0, eps):
    k = 0
    x_k = x0
    x_k_minus_1 = np.zeros(len(x_k))
    x_k_minus_n = {
        0: 0,
        1: 0,
        2: 0
    }
    s_last = {
        0: 0,
        1: 0,
        2: 0
    }
    s = {
        0: lambda x_k: np.matrix(-sf.Gradient(f, x_k)),
        1: lambda x_k: -sf.Gradient(f, x_k) + 
		sf.gamma_non_kvad(f, x_k, x_k, x_k_minus_n[0])*s_last[0],
        2: lambda x_k: -sf.Gradient(f, x_k) + 
		sf.gamma_non_kvad(f, x_k, x_k, x_k_minus_n[1])*s_last[1] + 
		sf.gamma_non_kvad(f,x_k,x_k_minus_n[1],x_k_minus_n[0])*s_last[0],
        3: lambda x_k: -sf.Gradient(f, x_k) + 
		sf.gamma_non_kvad(f,x_k,x_k,x_k_minus_n[2])*s_last[2]+ 
		sf.gamma_non_kvad(f,x_k,x_k_minus_n[2],x_k_minus_n[1])*s_last[1]+ 
		sf.gamma_non_kvad(f,x_k,x_k_minus_n[1],x_k_minus_n[0])*s_last[0]
    }
    x_points = [x_k];   

    while not sf.BreakCriterion(f, x_k, x_k_minus_1, eps):
        if k > 0:
            x_k_minus_1 = x_k_minus_n.get(k-1, x_k_minus_n.get(2))
            
        s_k = np.array(s.get(k, s.get(3))(x_k)).flatten()        
        if k < 3:
            s_last[k] = s_k

            x_k_minus_n[k] = x_k
        else:
            s_last[0] = s_last[1]
            s_last[1] = s_last[2]
            s_last[2] = s_k

            x_k_minus_n[0] = x_k_minus_n[1]
            x_k_minus_n[1] = x_k_minus_n[2]
            x_k_minus_n[2] = x_k
        
        beta_k = sf.findStep(f, x_k, s_k)
        x_k1 = x_k + beta_k*s_k
        #print(sf.f_at_point(f, x_k, True))
        x_k = x_k1
        k = k + 1
        x_points.append(x_k)
        
    f_star = sf.f_at_point(f, x_k, True)
    return {'x_star':x_k,'f_star':f_star,'k':k,'x_points':x_points}

def threeStepsCGM(f, x0, eps):
    k = 0
    x_k = x0
    x_k_minus_1 = np.zeros(len(x_k))
    s_last = {
        0: 0,
        1: 0
    }
    s = {
        0: lambda x_k: np.matrix(-sf.Gradient(f, x_k)),
        1: lambda x_k: -sf.Gradient(f, x_k) + 
		sf.gamma(f, x_k, s_last[0])*s_last[0],
        2: lambda x_k: -sf.Gradient(f, x_k) + 
		sf.gamma(f, x_k, s_last[1])*s_last[1] + 
		sf.gamma(f, x_k, s_last[0])*s_last[0]
    }
    x_points = [x_k];
    
    while not sf.BreakCriterion(f, x_k, x_k_minus_1, eps):
        s_k = np.array(s.get(k, s.get(2))(x_k)).flatten()
        if k < 2:
            s_last[k] = s_k
        else:
            s_last[0] = s_last[1]
            s_last[1] = s_k
        
        beta_k = sf.findStep(f, x_k, s_k) 
        x_k1 = x_k + beta_k*s_k
        x_k_minus_1 = x_k
        x_k = x_k1
        k = k + 1
        x_points.append(x_k)

    f_star = sf.f_at_point(f, x_k, True)   
    return {'x_star':x_k,'f_star':f_star,'k':k,'x_points':x_points}    
    
def nonQvadThreeStepsCGM(f, x0, eps):
    k = 0
    x_k = x0
    x_k_minus_1 = np.zeros(len(x_k))
    x_k_minus_n = {
        0: 0,
        1: 0
    }
    s_last = {
        0: 0,
        1: 0
    }
    s = {
        0: lambda x_k: np.matrix(-sf.Gradient(f, x_k)),
        1: lambda x_k: -sf.Gradient(f, x_k) + 
		sf.gamma_non_kvad(f, x_k, x_k, x_k_minus_n[0])*s_last[0],
        2: lambda x_k: -sf.Gradient(f, x_k) + 
		sf.gamma_non_kvad(f,x_k,x_k,x_k_minus_n[1])*s_last[1] + 
		sf.gamma_non_kvad(f,x_k,x_k_minus_n[1],x_k_minus_n[0])*s_last[0]
    }
    x_points = [x_k];   

    while not sf.BreakCriterion(f, x_k, x_k_minus_1, eps):
        if k > 0:
            x_k_minus_1 = x_k_minus_n.get(k-1, x_k_minus_n.get(1))
            
        s_k = np.array(s.get(k, s.get(2))(x_k)).flatten()        
        if k < 2:
            s_last[k] = s_k

            x_k_minus_n[k] = x_k
        else:
            s_last[0] = s_last[1]
            s_last[1] = s_k

            x_k_minus_n[0] = x_k_minus_n[1]
            x_k_minus_n[1] = x_k
        
        beta_k = sf.findStep(f, x_k, s_k)
        x_k1 = x_k + beta_k*s_k
        #print(sf.f_at_point(f, x_k, True))
        x_k = x_k1
        k = k + 1
        x_points.append(x_k)
        
    f_star = sf.f_at_point(f, x_k, True)
    return {'x_star':x_k,'f_star':f_star,'k':k,'x_points':x_points}    
