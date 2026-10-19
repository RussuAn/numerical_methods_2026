from math import erf
import numpy as np

def load(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

def primitive(x):
    return 50 * x - 240 / np.pi * np.cos(np.pi * x / 12) + 5 * np.sqrt(np.pi) / (2 * np.sqrt(0.2)) * erf(np.sqrt(0.2) * (x - 12))

def exact_integral(a, b):
    return primitive(b) - primitive(a)

def simpson(f, a, b, n):
    if n < 2 or n % 2 != 0:
        raise ValueError("Число розбиттів має бути додатним і парним")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))

def runge_romberg(i_n, i_half):
    return i_n + (i_n - i_half) / 15

def aitken(i_n, i_half, i_quarter):
    denominator = 2 * i_half - i_n - i_quarter
    if denominator == 0:
        raise ValueError("Нульовий знаменник у формулі Ейткена")
    return (i_half**2 - i_n * i_quarter) / denominator

def accuracy_order(i_n, i_half, i_quarter):
    if i_half == i_n or i_quarter == i_half:
        raise ValueError("Неможливо визначити порядок точності")
    ratio = abs((i_quarter - i_half) / (i_half - i_n))
    return np.log(ratio) / np.log(2)

def adaptive_simpson(f, a, b, delta, max_depth=50):
    if delta <= 0 or a >= b:
        raise ValueError("Потрібно delta > 0 та a < b")
    middle = (a + b) / 2
    fa, fm, fb = f(a), f(middle), f(b)
    count = 3
    first = (b - a) / 6 * (fa + 4 * fm + fb)

    def divide(left, right, fl, fc, fr, coarse, tolerance, depth):
        nonlocal count
        center = (left + right) / 2
        x1 = (left + center) / 2
        x3 = (center + right) / 2
        if depth >= max_depth or x1 == left or x3 == right:
            raise RuntimeError("Досягнуто межі поділу відрізка")
        f1, f3 = f(x1), f(x3)
        count += 2
        s_left = (center - left) / 6 * (fl + 4 * f1 + fc)
        s_right = (right - center) / 6 * (fc + 4 * f3 + fr)
        fine = s_left + s_right
        if abs(coarse - fine) <= tolerance:
            return fine
        result_left = divide(left, center, fl, f1, fc, s_left, tolerance / 2, depth + 1)
        result_right = divide(center, right, fc, f3, fr, s_right, tolerance / 2, depth + 1)
        return result_left + result_right
    value = divide(a, b, fa, fm, fb, first, delta, 0)
    return value, count