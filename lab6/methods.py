from decimal import Decimal, localcontext


def read_matrix(filename):
    with open(filename, encoding="utf-8") as file:
        return [
            list(map(float, line.split()))
            for line in file
            if line.strip()
        ]


def read_vector(filename):
    with open(filename, encoding="utf-8") as file:
        return [float(value) for value in file.read().split()]


def write_matrix(filename, matrix):
    with open(filename, "w", encoding="utf-8") as file:
        for row in matrix:
            file.write(" ".join(str(value) for value in row) + "\n")


def write_vector(filename, vector):
    with open(filename, "w", encoding="utf-8") as file:
        for value in vector:
            file.write(str(value) + "\n")


def write_lu(filename, L, U):
    with open(filename, "w", encoding="utf-8") as file:
        file.write("L:\n")
        for row in L:
            file.write(" ".join(str(value) for value in row) + "\n")

        file.write("\nU:\n")
        for row in U:
            file.write(" ".join(str(value) for value in row) + "\n")


def matrix_vector(A, x):
    result = []

    for row in A:
        total = 0.0
        for j in range(len(x)):
            total += row[j] * x[j]
        result.append(total)

    return result


def vector_norm(vector):
    return max(abs(value) for value in vector)


def lu_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        U[i][i] = 1.0

    for k in range(n):
        for i in range(k, n):
            total = 0.0
            for j in range(k):
                total += L[i][j] * U[j][k]

            L[i][k] = A[i][k] - total

        if L[k][k] == 0:
            raise ValueError(
                "Нульовий опорний елемент: цей LU-розклад неможливий."
            )

        for i in range(k + 1, n):
            total = 0.0
            for j in range(k):
                total += L[k][j] * U[j][i]

            U[k][i] = (A[k][i] - total) / L[k][k]

    return L, U


def solve_lu(L, U, b):
    n = len(b)
    z = [0.0] * n
    x = [0.0] * n

    for i in range(n):
        total = 0.0
        for j in range(i):
            total += L[i][j] * z[j]

        z[i] = (b[i] - total) / L[i][i]

    for i in range(n - 1, -1, -1):
        total = 0.0
        for j in range(i + 1, n):
            total += U[i][j] * x[j]

        x[i] = z[i] - total

    return x


def residual(A, x, b):
    with localcontext() as context:
        context.prec = 40
        r = []

        for i in range(len(b)):
            total = Decimal(0)

            for j in range(len(x)):
                total += Decimal(A[i][j]) * Decimal(x[j])

            r.append(float(Decimal(b[i]) - total))

    return r


def refine_solution(A, b, L, U, x, eps):
    x = x.copy()
    history = []

    for iteration in range(1, 101):
        r = residual(A, x, b)
        dx = solve_lu(L, U, r)

        x = [x[i] + dx[i] for i in range(len(x))]

        correction_norm = vector_norm(dx)
        residual_norm = vector_norm(residual(A, x, b))

        history.append(
            (iteration, correction_norm, residual_norm)
        )

        if correction_norm <= eps and residual_norm <= eps:
            return x, history, True

    return x, history, False