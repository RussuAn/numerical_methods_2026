import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
COEFFICIENTS = [1.0, -4.0, 6.0, -4.0]

def save_coefficients():
    DATA.mkdir(exist_ok=True)
    with open(DATA / 'polynomial_coefficients.txt', 'w', encoding='utf-8') as output:
        for c in COEFFICIENTS:
            output.write(f'{c:.15f}\n')

def read_coefficients(filename=DATA / 'polynomial_coefficients.txt'):
    with open(filename, encoding='utf-8') as source:
        return [float(line.strip()) for line in source if line.strip()]

def horner(coefficients, x):
    b = coefficients[0]
    for c in coefficients[1:]:
        b = b * x + c
    return b

def horner_with_derivative(coefficients, x):
    value = coefficients[0]
    derivative = 0.0
    for c in coefficients[1:]:
        derivative = derivative * x + value
        value = value * x + c
    return value, derivative

def newton_horner(coefficients, x0=2.3, eps=1e-10, max_iter=1000):
    x = x0
    history = []
    for k in range(1, max_iter + 1):
        value, derivative = horner_with_derivative(coefficients, x)
        if abs(derivative) < 1e-14:
            raise ZeroDivisionError('Нульова похідна многочлена')
        new = x - value / derivative
        residue = abs(horner(coefficients, new))
        diff = abs(new - x)
        history.append((k, new, residue, diff))
        if residue < eps and diff < eps:
            return new, k, history
        x = new
    raise RuntimeError('Ньютон–Горнер не збігся')

def lin(coefficients, alpha0=0.8, beta0=1.2, eps=1e-10, max_iter=1000):
    if len(coefficients) != 4:
        raise ValueError('Для методу Ліна потрібен кубічний многочлен')
    a3, a2, a1, a0 = coefficients
    p = -2 * alpha0
    q = alpha0 * alpha0 + beta0 * beta0
    history = []
    for k in range(1, max_iter + 1):
        b3 = a3
        b2 = a2 - p * b3
        if abs(b2) < 1e-14:
            raise ZeroDivisionError('Нульове b2 у методі Ліна')
        q_new = a0 / b2
        p_new = (a1 - b3 * q_new) / b2
        new_alpha = -p_new / 2
        beta_square = q_new - new_alpha * new_alpha
        if beta_square <= 0:
            raise ValueError('Проміжний квадратний множник не дає комплексних коренів')
        new_beta = math.sqrt(beta_square)
        r1 = a1 - b3 * q_new - b2 * p_new
        r0 = a0 - b2 * q_new
        actual_b2 = a2 - a3 * p_new
        actual_r1 = a1 - a3 * q_new - actual_b2 * p_new
        actual_r0 = a0 - actual_b2 * q_new
        remainder = max(abs(actual_r0), abs(actual_r1))
        old_alpha = -p / 2
        old_beta_sq = q - old_alpha * old_alpha
        old_beta = math.sqrt(old_beta_sq) if old_beta_sq > 0 else 0
        diff = max(abs(new_alpha - old_alpha), abs(new_beta - old_beta))
        history.append((k, new_alpha, new_beta, remainder, diff))
        if remainder < eps and diff < eps:
            z = complex(new_alpha, new_beta)
            return (z, z.conjugate()), k, history
        p, q = p_new, q_new
    raise RuntimeError('Метод Ліна не збігся')