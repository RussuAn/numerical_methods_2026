from pathlib import Path
import numpy as np

n = 100

folder = Path(__file__).resolve().parent / "data"
folder.mkdir(exist_ok=True)

np.random.seed(42)
A = np.random.uniform(-0.01, 0.01, (n, n))

for i in range(n):
    s = 0
    for j in range(n):
        if i != j:
            s += abs(A[i, j])
    A[i, i] = s + 0.05

x = np.full(n, 2.5)

b = np.zeros(n)

for i in range(n):
    for j in range(n):
        b[i] += A[i, j] * x[j]

np.savetxt(folder / "matrix_A.txt", A)
np.savetxt(folder / "vector_b.txt", b)

print("Матрицю A збережено.")
print("Вектор b збережено.")