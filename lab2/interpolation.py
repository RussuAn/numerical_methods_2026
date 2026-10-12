import numpy as np

def omega_k(x, nodes, k):
    prod = 1.0
    for i in range(k + 1):
        prod *= (x - nodes[i])
    return prod

def divided_differences(x, y):
    n = len(x)
    T = np.zeros((n, n))
    T[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            T[i, j] = (T[i+1, j-1] - T[i, j-1]) / (x[i+j] - x[i])
    return T

def newton_eval(x_nodes, coeffs, x_val):
    res = coeffs[0]
    for k in range(1, len(coeffs)):
        res += coeffs[k] * omega_k(x_val, x_nodes, k - 1)
    return res

def lagrange_eval(xn, yn, xv):
    s = 0.0
    n = len(xn)
    for i in range(n):
        p = 1.0
        for j in range(n):
            if i != j:
                p *= (xv - xn[j]) / (xn[i] - xn[j])
        s += yn[i] * p
    return s

def finite_differences(y):
    n = len(y)
    T = np.zeros((n, n))
    T[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            T[i, j] = T[i+1, j-1] - T[i, j-1]
    return T

def factorial_eval(x0, h, fd_row0, xv):
    t = (xv - x0) / h
    res = fd_row0[0]
    prod = 1.0
    fact = 1.0
    for k in range(1, len(fd_row0)):
        prod *= (t - (k - 1))
        fact *= k
        res += (fd_row0[k] / fact) * prod
    return res

def runge_function(x):
    return 1.0 / (1.0 + 25.0 * x**2)