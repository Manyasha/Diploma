from sympy import diff, symbols, Symbol, cos, sin
import numpy as np
from scipy import optimize

x, y = symbols('x y')
r = diff(cos(x) + 1j*sin(y), x)
g = diff(cos(x) + 1j*sin(y), y)
print(r)
print(g)
def f(x):
    return x**3 + x**2
print(type(f(x)))
k = diff(f(x),x)
print(k)

print(k.subs({x:1}).n())


s1 = 'g'
s2 = str(1)
print(s1 + s2)
x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
def d():
    #x1 = symbols('x1')
    return x1**3 + x2**2;
def f1():
    return 100*(x2 - x1**2)**2 + (1 - x1)**2
def gradient(f, x0):
    print('---------')
    #x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
    x = np.array([x1, x2, x3, x4])
    #n = x0.size
    n = len(x0)
    grad = np.zeros(n)
    for i in range(n):
        grad[i] = diff(f, x[i]).subs(x0).n()
    return grad

#print(gradient(d(), np.array([2], dtype=float)))
print(gradient(f1(),{x1: -1.2, x2: 1}))


def f(x):
	return 5*(1-x[0]) + x[1]
print(optimize.fmin_cg(f, [2,2]))
