import numpy as np


def form_matrix(x, m):
    A = np.zeros((m + 1, m + 1), dtype=float)
    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = np.sum(x ** (i + j))
    return A


def form_vector(x, y, m):
    b = np.zeros(m + 1, dtype=float)
    for i in range(m + 1):
        b[i] = np.sum(y * (x ** i))
    return b


def gauss_solve(A_in, b_in):
    n = len(b_in)
    A = A_in.copy().astype(float)
    b = b_in.copy().astype(float)
    for k in range(n - 1):
        max_row = k + np.argmax(np.abs(A[k:, k]))
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    x_sol = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.sum(A[i, i + 1:] * x_sol[i + 1:])) / A[i, i]
    return x_sol


def eval_polynomial(x_vals, coef):
    res = np.zeros_like(x_vals, dtype=float)
    for p, c in enumerate(coef):
        res += c * (x_vals ** p)
    return res


def calc_variance(y_true, y_approx):
    return np.sqrt(np.mean((y_true - y_approx) ** 2))