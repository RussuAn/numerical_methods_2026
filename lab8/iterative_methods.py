import math
from functions import f, df, d2f

EPS = 1e-10
MAX_ITER = 1000

def finish(x, prev, it, history, eps):
    residue = abs(f(x))
    diff = abs(x - prev)
    history.append((it, x, residue, diff))
    return residue < eps and diff < eps

def simple_iteration(x0, eps=EPS, max_iter=MAX_ITER):
    slope = df(x0)
    if abs(slope) < 1e-14:
        raise ValueError('Похідна в початковій точці надто мала')
    lam = -1.0 / slope
    x = x0
    history = []
    for k in range(1, max_iter + 1):
        new = x + lam * f(x)
        if finish(new, x, k, history, eps):
            return new, k, history
        x = new
    raise RuntimeError('Проста ітерація не збіглася')

def newton(x0, eps=EPS, max_iter=MAX_ITER):
    x = x0
    history = []
    for k in range(1, max_iter + 1):
        derivative = df(x)
        if abs(derivative) < 1e-14:
            raise ZeroDivisionError('Нульова похідна в методі Ньютона')
        new = x - f(x) / derivative
        if finish(new, x, k, history, eps):
            return new, k, history
        x = new
    raise RuntimeError('Метод Ньютона не збігся')

def chebyshev(x0, eps=EPS, max_iter=MAX_ITER):
    x = x0
    history = []
    for k in range(1, max_iter + 1):
        d = df(x)
        if abs(d) < 1e-14:
            raise ZeroDivisionError('Нульова похідна в методі Чебишева')
        fx = f(x)
        new = x - fx / d - fx * fx * d2f(x) / (2 * d ** 3)
        if finish(new, x, k, history, eps):
            return new, k, history
        x = new
    raise RuntimeError('Метод Чебишева не збігся')

def secant(x0, x1, eps=EPS, max_iter=MAX_ITER):
    history = []
    for k in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)
        if abs(f1 - f0) < 1e-15:
            raise ZeroDivisionError('Збіглися значення функції в методі хорд')
        new = x1 - f1 * (x1 - x0) / (f1 - f0)
        if finish(new, x1, k, history, eps):
            return new, k, history
        x0, x1 = x1, new
    raise RuntimeError('Метод хорд не збігся')

def parabola(x0, x1, x2, eps=EPS, max_iter=MAX_ITER):
    history = []
    for k in range(1, max_iter + 1):
        h0 = x1 - x0
        h1 = x2 - x1
        if h0 == 0 or h1 == 0:
            raise ZeroDivisionError('Збіглися вузли методу парабол')
        d0 = (f(x1) - f(x0)) / h0
        d1 = (f(x2) - f(x1)) / h1
        a = (d1 - d0) / (h1 + h0)
        b = d1 + h1 * a
        c = f(x2)
        discriminant = b * b - 4 * a * c
        if discriminant < -1e-12:
            raise ValueError('Комплексна поправка при уточненні дійсного кореня')
        root = math.sqrt(max(0.0, discriminant))
        denominator = b + root if abs(b + root) > abs(b - root) else b - root
        if abs(denominator) < 1e-15:
            raise ZeroDivisionError('Нульовий знаменник методу парабол')
        new = x2 - 2 * c / denominator
        if finish(new, x2, k, history, eps):
            return new, k, history
        x0, x1, x2 = x1, x2, new
    raise RuntimeError('Метод парабол не збігся')

def inverse_interpolation(x0, x1, x2, eps=EPS, max_iter=MAX_ITER):
    history = []
    for k in range(1, max_iter + 1):
        y0, y1, y2 = f(x0), f(x1), f(x2)
        denominators = ((y0-y1)*(y0-y2), (y1-y0)*(y1-y2), (y2-y0)*(y2-y1))
        if any(abs(d) < 1e-27 for d in denominators):
            if abs(y2 - y1) < 1e-15:
                raise ZeroDivisionError('Нульовий знаменник зворотної інтерполяції')
            new = x2 - y2 * (x2-x1) / (y2-y1)
        else:
            new = (x0*y1*y2/denominators[0] +
                   x1*y0*y2/denominators[1] +
                   x2*y0*y1/denominators[2])
        if finish(new, x2, k, history, eps):
            return new, k, history
        x0, x1, x2 = x1, x2, new
    raise RuntimeError('Зворотна інтерполяція не збіглася')

def run_all_for_interval(interval, eps=EPS):
    left, right = interval
    center = (left + right) / 2
    seeds = (left, center, right)
    methods = {
        'Проста ітерація': simple_iteration(center, eps),
        'Ньютона': newton(center, eps),
        'Чебишева': chebyshev(center, eps),
        'Хорд': secant(left, right, eps),
        'Парабол': parabola(*seeds, eps=eps),
        'Зворотна інтерполяція': inverse_interpolation(*seeds, eps=eps),
    }
    return methods