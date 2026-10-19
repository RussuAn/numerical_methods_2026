import os
import numpy as np
from methods import load, exact_integral, simpson, runge_romberg, aitken, accuracy_order, adaptive_simpson
import plots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "graphs")
os.makedirs(OUT_DIR, exist_ok=True)

a = 0
b = 24
epsilon = 1e-12

ns = np.arange(10, 1001, 2)
deltas = 10.0 ** np.arange(-2, -13, -1)

exact = exact_integral(a, b)
print(f"Аналітичне значення інтегралу I0 = {exact:.15f}")

values = np.array([simpson(load, a, b, int(n)) for n in ns])
errors = np.abs(values - exact)

suitable = np.where(errors <= epsilon)[0]

if len(suitable) == 0:
    raise RuntimeError("На перевіреній сітці заданої точності не досягнуто")

index = suitable[0]
n_opt = int(ns[index])

indices = np.linspace(0, len(ns) - 1, 15, dtype=int)
indices = np.unique(np.append(indices, index))

print("\nТАБЛИЦЯ 1. ЗАЛЕЖНІСТЬ ПОХИБКИ ВІД N (ВИБРАНІ ЗНАЧЕННЯ)")
print("-" * 60)
print(f"{'N':>6} | {'I(N)':>22} | {'Абсолютна похибка':>24}")
print("-" * 60)

for i in indices:
    print(f"{ns[i]:6d} | {values[i]:22.15f} | {errors[i]:24.9e}")

print(f"\nЗадана точність: {epsilon:.6e}")
print(f"N_opt = {n_opt}")
print(f"Отримана похибка: {errors[index]:.9e}")

n0 = max(8, int(np.floor(n_opt / 10 / 8 + 0.5)) * 8)

n_values = [n0, n0 // 2, n0 // 4]
i_values = [simpson(load, a, b, n) for n in n_values]
i_n, i_half, i_quarter = i_values

print(f"\nN0 = {n0}")

print("\nТАБЛИЦЯ 2. ОБЧИСЛЕННЯ ДЛЯ N0, N0/2, N0/4")
print("-" * 60)
print(f"{'N':>6} | {'I(N)':>22} | {'Абсолютна похибка':>24}")
print("-" * 60)

for n, value in zip(n_values, i_values):
    error = abs(value - exact)
    print(f"{n:6d} | {value:22.15f} | {error:24.9e}")

rr = runge_romberg(i_n, i_half)
ai = aitken(i_n, i_half, i_quarter)
order = accuracy_order(i_n, i_half, i_quarter)

names = ["Сімпсон (N0)", "Рунге–Ромберг", "Ейткен"]
results = [i_n, rr, ai]
result_errors = np.abs(np.array(results) - exact)

print("\nТАБЛИЦЯ 3. ПОРІВНЯННЯ МЕТОДІВ")
print("-" * 90)
print(f"{'Метод':<18} | {'Інтеграл':>22} | {'Похибка':>24} | {'eps0 / eps':>12}")
print("-" * 90)

for name, value, error in zip(names, results, result_errors):
    if error != 0:
        ratio = f"{result_errors[0] / error:.6f}"
    else:
        ratio = "не визначено"
    print(f"{name:<18} | {value:22.15f} | {error:24.9e} | {ratio:>12}")

print(f"\nОцінений порядок точності: p = {order:.9f}")

adaptive_errors = []
counts = []

print("\nТАБЛИЦЯ 4. АДАПТИВНИЙ АЛГОРИТМ")
print("-" * 90)
print(f"{'δ':>20} | {'Інтеграл':>22} | {'Похибка':>24} | {'Виклики f':>10}")
print("-" * 90)

for delta in deltas:
    value, count = adaptive_simpson(load, a, b, delta)
    error = abs(value - exact)
    adaptive_errors.append(error)
    counts.append(count)
    print(f"{delta:20.6e} | {value:22.15f} | {error:24.9e} | {count:10d}")

plots.plot_function(load, a, b, OUT_DIR)
plots.plot_errors(ns, errors, n_opt, epsilon, OUT_DIR)
plots.plot_comparison(names, result_errors, OUT_DIR)
plots.plot_adaptive(deltas, adaptive_errors, counts, OUT_DIR)