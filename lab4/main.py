import os
import numpy as np
from methods import exact_derivative, central_difference, absolute_error, runge_romberg, aitken, accuracy_order, search_best_step
import plots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "graphs")
os.makedirs(OUT_DIR, exist_ok=True)

t0 = 1
h = 0.001
q = 2
p = 2

exact = exact_derivative(t0)

print(f"Точне значення M'(t0) = {exact:.12f}")

steps = 10.0 ** np.arange(-20, 4)
best_h, best_error, errors = search_best_step(t0, steps)

print("\n" + "-" * 65)
print("ТАБЛИЦЯ 1. ЗАЛЕЖНІСТЬ ПОХИБКИ ВІД КРОКУ")
print("-" * 65)
print(f"{'Крок h':>14} | {'D(h)':>19} | {'Абсолютна похибка':>21}")
print("-" * 65)

for i in range(len(steps)):
    step = steps[i]
    value = central_difference(t0, step)
    print(f"{step:>14.6e} | {value:>19.12f} | {errors[i]:>21.9e}")

print(f"\nНайкращий крок на перевіреній сітці: h0 = {best_h:.9e}")
print(f"Найменша похибка на сітці: {best_error:.9e}")

h_values = np.array([h, q * h, q**2 * h])
d_values = central_difference(t0, h_values)
d_h, d_2h, d_4h = d_values

print("\n" + "-" * 65)
print("ТАБЛИЦЯ 2. ЧИСЕЛЬНЕ ДИФЕРЕНЦІЮВАННЯ")
print("-" * 65)
print(f"{'Крок h':>14} | {'Значення похідної':>19} | {'Абсолютна похибка':>21}")
print("-" * 65)

for step, value in zip(h_values, d_values):
    error = absolute_error(value, exact)
    print(f"{step:>14.6g} | {value:>19.12f} | {error:>21.9e}")

rr = runge_romberg(d_h, d_2h, q, p)
ai = aitken(d_h, d_2h, d_4h)
order = accuracy_order(d_h, d_2h, d_4h, q)

print("\n" + "-" * 45)
print("МЕТОД РУНГЕ–РОМБЕРГА")
print("-" * 45)
print(f"Уточнене значення: {rr:.12f}")
print(f"Абсолютна похибка: {absolute_error(rr, exact):.9e}")

print("\n" + "-" * 45)
print("МЕТОД ЕЙТКЕНА")
print("-" * 45)
print(f"Уточнене значення: {ai:.12f}")
print(f"Оцінений порядок точності: p = {order:.9f}")
print(f"Абсолютна похибка: {absolute_error(ai, exact):.9e}")

results = [
    ("Аналітична похідна", exact),
    ("Центральна різниця (h)", d_h),
    ("Центральна різниця (2h)", d_2h),
    ("Центральна різниця (4h)", d_4h),
    ("Рунге–Ромберг", rr),
    ("Ейткен", ai)
]

print("\n" + "-" * 73)
print("ТАБЛИЦЯ 3. ПОРІВНЯННЯ РЕЗУЛЬТАТІВ")
print("-" * 73)
print(f"{'Метод':<27} | {'Значення похідної':>19} | {'Абсолютна похибка':>21}")
print("-" * 73)

for name, value in results:
    error = absolute_error(value, exact)
    print(f"{name:<27} | {value:>19.12f} | {error:>21.9e}")

plots.plot_function(OUT_DIR)
plots.plot_step_error(steps, errors, best_h, best_error, OUT_DIR)
plots.plot_comparison([d_h, rr, ai], exact, OUT_DIR)