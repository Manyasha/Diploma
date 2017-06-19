from cmath import e
from sympy import symbols

x1, x2, x3, x4 = symbols('x1 x2 x3 x4')

a = [0.0, 0.000428, 0.001, 0.00161, 0.00209, 0.00348, 0.00525]
b = [7.391, 11.18, 16.44, 16.2, 22.2, 24.02, 31.32]
c = [1.5, 2.25, 2.625]

f = {
    1: 100*(x2 - x1**2)**2 + (1 - x1)**2,
    2: (x1 + 10*x2)**2 + 5*(x3 - x4)**2 + (x2 - 2*x3)**4 + 10*(x1 - x4)**4,
    3: (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2,
    4: (x1**2 + 12*x2 - 1)**2 + (49*x1**2 + 49*x2**2 + 84*x1 + 2324*x2 - 681)**2,
    5: 100*(x3 - ((x1 + x2)/2)**2)**2 + (1 - x1)**2 + (1 - x2)**2,
    6: 10**4 * sum([((x1**2 + x2**2 * a[i] + x3**2 * a[i]**2)/(1 + x4**2 * a[i]) - b[i]) / b[i] for i in range(7)]),
    7: sum([(c[i] - x1*(1 - x2**(i + 1)))**2 for i in range(3)]),
    8: (x2 - x1**2)**2 + (1 - x1)**2,
    9: (x2 - x1**2)**2 + 100*(1 - x1)**2,
    10: 100*(x2 - x1**3)**2 + (1 - x1)**2,
    11: 100*(x2 - x1**2)**2 + (1 - x1)**2 + 90*(x4 - x3**2) + (1 - x3)**3 + 10.1*((x2 - 1)**2 + (x4 - 1)**2) + 19.8*(x2 - 1)*(x4 - 1),
    12: (x1 + 40*x2)**2 + 5*(x3 - x4)**2 + (x2 - 2*x3)**4 + 10*(x1 - x4)**4,
    13: -x1**2 * e**(1 - x1**2 - 20.25*(x1 - x2)**2)
}

x0 = {
    1: [-1.2, 1],
    2: [[3, -1, 0, 1], [1, 1, 1, 1]],
    3: [1, 1],
    4: [1, 1],
    5: [-1.2, 2, 0],
    6: [2.7, 90, 1500, 10],
    7: [2, 0.2],
    8: [-1.2, 1],
    9: [-1.2, 1],
    10: [-1.2, 1],
    11: [-3, -1, -3, -1],
    12: [-3, -1, 0, 1],
    13: [0.1, 0.1]
}

x_star = {
    1: [1, 1],
    2: [0, 0, 0, 0],  
    3: [[3.58443, -1.84813], [3, 2]],
    4: [[0.28581, 0.27936], [-21.026653, -36.7660090]],
    5: [1, 1, 1],
    6: [2.714, 140.4, 1707, 31.51],
    7: [3, 0.5],
    8: [1, 1],
    9: [1, 1],
    10: [1, 1],
    11: [1, 1, 1, 1],
    12: [0, 0, 0, 0],
    13: [1.9, 0]
}

f_star = {
    1: 0,
    2: 0, 
    3: 0,
    4: [5.9225, 0],
    5: 0,
    6: 70000,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0
}

