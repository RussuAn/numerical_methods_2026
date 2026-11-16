from pathlib import Path
import numpy as np

from methods import (
    read_matrix, read_vector, matrix_vector,
    vector_norm, matrix_norm,
    simple_iteration, jacobi, seidel
)

from plots import (
    plot_convergence, plot_iterations, plot_errors
)

folder = Path(__file__).resolve().parent
data = folder / "data"
plots = folder / "plots"
plots.mkdir(exist_ok=True)

A = read_matrix(data / "matrix_A.txt")
b = read_vector(data / "vector_b.txt")

n = len(b)
eps = 1e-14

x_exact = np.full(n, 2.5)
x0 = np.ones(n)

print("Розмірність системи:", n)
print("Точний розв'язок: x_i = 2.5")
print("Початкове наближення: x_i = 1.0")
print("Норма матриці:", matrix_norm(A))
print("Задана точність:", eps)

x_simple, k_simple, h_simple, d_simple = simple_iteration(
    A, b, x0, eps
)

x_jacobi, k_jacobi, h_jacobi, d_jacobi = jacobi(
    A, b, x0, eps
)

x_seidel, k_seidel, h_seidel, d_seidel = seidel(
    A, b, x0, eps
)

error_simple = vector_norm(x_simple - x_exact)
error_jacobi = vector_norm(x_jacobi - x_exact)
error_seidel = vector_norm(x_seidel - x_exact)

r_simple = vector_norm(matrix_vector(A, x_simple) - b)
r_jacobi = vector_norm(matrix_vector(A, x_jacobi) - b)
r_seidel = vector_norm(matrix_vector(A, x_seidel) - b)

print("\nМетод простої ітерації")
print("Перші 10 компонент:", x_simple[:10])
print("Кількість ітерацій:", k_simple)
print("Похибка:", error_simple)
print("Різниця наближень:", d_simple)
print("Норма нев'язки:", r_simple)

print("\nМетод Якобі")
print("Перші 10 компонент:", x_jacobi[:10])
print("Кількість ітерацій:", k_jacobi)
print("Похибка:", error_jacobi)
print("Різниця наближень:", d_jacobi)
print("Норма нев'язки:", r_jacobi)

print("\nМетод Зейделя")
print("Перші 10 компонент:", x_seidel[:10])
print("Кількість ітерацій:", k_seidel)
print("Похибка:", error_seidel)
print("Різниця наближень:", d_seidel)
print("Норма нев'язки:", r_seidel)

print("\nПеревірка точності")

print("Проста ітерація:",
      d_simple < eps and r_simple <= eps)

print("Якобі:",
      d_jacobi < eps and r_jacobi <= eps)

print("Зейделя:",
      d_seidel < eps and r_seidel <= eps)

plot_convergence(h_simple, h_jacobi, h_seidel, plots)

plot_iterations(
    [k_simple, k_jacobi, k_seidel], plots
)

plot_errors(
    [error_simple, error_jacobi, error_seidel], plots
)