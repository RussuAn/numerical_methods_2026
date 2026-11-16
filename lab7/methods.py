import numpy as np

def read_matrix(filename):
    return np.loadtxt(filename)

def read_vector(filename):
    return np.loadtxt(filename)

def matrix_vector(A, x):
    n = len(x)
    result = np.zeros(n)

    for i in range(n):
        for j in range(n):
            result[i] += A[i, j] * x[j]

    return result


def vector_norm(x):
    return max(abs(x))

def matrix_norm(A):
    n = len(A)
    max_sum = 0

    for i in range(n):
        s = 0
        for j in range(n):
            s += abs(A[i, j])

        if s > max_sum:
            max_sum = s

    return max_sum


def simple_iteration(A, b, x0, eps, max_iter=10000):
    n = len(b)
    x = x0.copy()
    tau = 1 / matrix_norm(A)
    history = []

    for k in range(max_iter):
        Ax = matrix_vector(A, x)
        x_new = np.zeros(n)

        for i in range(n):
            x_new[i] = x[i] - tau * (Ax[i] - b[i])

        difference = vector_norm(x_new - x)
        residual = vector_norm(matrix_vector(A, x_new) - b)

        history.append(residual)

        if difference < eps and residual <= eps:
            return x_new, k + 1, history, difference

        x = x_new.copy()

    raise RuntimeError("Метод простої ітерації не збігся")


def jacobi(A, b, x0, eps, max_iter=10000):
    n = len(b)
    x = x0.copy()
    history = []

    for k in range(max_iter):
        x_new = np.zeros(n)

        for i in range(n):
            s = 0

            for j in range(n):
                if i != j:
                    s += A[i, j] * x[j]

            x_new[i] = (b[i] - s) / A[i, i]

        difference = vector_norm(x_new - x)
        residual = vector_norm(matrix_vector(A, x_new) - b)

        history.append(residual)

        if difference < eps and residual <= eps:
            return x_new, k + 1, history, difference

        x = x_new.copy()

    raise RuntimeError("Метод Якобі не збігся")


def seidel(A, b, x0, eps, max_iter=10000):
    n = len(b)
    x = x0.copy()
    history = []

    for k in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            s1 = 0
            s2 = 0

            for j in range(i):
                s1 += A[i, j] * x_new[j]

            for j in range(i + 1, n):
                s2 += A[i, j] * x[j]

            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        difference = vector_norm(x_new - x)
        residual = vector_norm(matrix_vector(A, x_new) - b)

        history.append(residual)

        if difference < eps and residual <= eps:
            return x_new, k + 1, history, difference

        x = x_new.copy()

    raise RuntimeError("Метод Зейделя не збігся")