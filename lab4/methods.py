import numpy as np

def moisture(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def exact_derivative(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

def central_difference(t, h):
    return (moisture(t + h) - moisture(t - h)) / (2 * h)

def absolute_error(approximation, exact):
    return np.abs(approximation - exact)

def runge_romberg(d_h, d_qh, q=2, p=2):
    if q <= 1 or p <= 0:
        raise ValueError('Потрібно q > 1 і p > 0')
    return d_h + (d_h - d_qh) / (q**p - 1)

def aitken(d_h, d_qh, d_q2h):
    denominator = 2 * d_qh - d_q2h - d_h
    if np.isclose(denominator, 0, rtol=0, atol=1e-15):
        raise ValueError('Метод Ейткена: знаменник надто близький до нуля')
    return (d_qh**2 - d_h*d_q2h) / denominator

def accuracy_order(d_h, d_qh, d_q2h, q=2):
    if q <= 1 or d_qh == d_h or d_q2h == d_qh:
        raise ValueError('Неможливо визначити порядок точності')
    return np.log(abs((d_q2h - d_qh) / (d_qh - d_h))) / np.log(q)

def search_best_step(t, steps):
    exact = exact_derivative(t)
    approximations = central_difference(t, steps)
    errors = absolute_error(approximations, exact)
    index = int(np.argmin(errors))
    return float(steps[index]), float(errors[index]), errors