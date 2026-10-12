import os
import numpy as np
from data_processing import init_csv_file, read_data
from interpolation import (
    divided_differences,
    newton_eval,
    finite_differences,
    factorial_eval,
    lagrange_eval
)
import plots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data.csv")
OUT_DIR  = os.path.join(BASE_DIR, "графіки")
os.makedirs(OUT_DIR, exist_ok=True)

init_csv_file(CSV_PATH)
x_data, y_data = read_data(CSV_PATH)

print("Вхідні експериментальні дані:")
print("-" * 35)
print(f"{'Tasks (n)':>10} | {'Cost ($)':>12}")
print("-" * 35)
for xi, yi in zip(x_data, y_data):
    print(f"{xi:>10.0f} | {yi:>12.2f}")

DD = divided_differences(x_data, y_data)
newton_coeffs = list(DD[0, :])

print("\nТАБЛИЦЯ РОЗДІЛЕНИХ РІЗНИЦЬ:")
print("-" * 88)
print(f"{'i':>4} | {'f(xi)':>10} | {'1-й':>12} | {'2-й':>12} | {'3-й':>12} | {'4-й':>12}")
print("-" * 88)
for i in range(len(x_data)):
    row = f"{i:>4} |"
    for j in range(len(x_data) - i):
        row += f" {DD[i, j]:>12.4e} |"
    print(row)

print("\nКоефіцієнти многочлена Ньютона:")
for i, c in enumerate(newton_coeffs):
    print(f"c[{i}] = {c:>16.8e}")

K_uni = 5
x_uni = np.linspace(x_data[0], x_data[-1], K_uni)
y_uni = np.array([newton_eval(x_data, newton_coeffs, xi) for xi in x_uni])
h_uni = x_uni[1] - x_uni[0]
FD = finite_differences(y_uni)

print(f"\nТАБЛИЦЯ СКІНЧЕННИХ РІЗНИЦЬ (h = {h_uni:.1f}):")
print("-" * 75)
for i in range(K_uni):
    row = f"{i:>4} |"
    for j in range(K_uni - i):
        row += f" {FD[i, j]:>12.4e} |"
    print(row)

x_target = 15000.0
pred_newton    = newton_eval(x_data, newton_coeffs, x_target)
pred_factorial = factorial_eval(x_uni[0], h_uni, FD[0, :], x_target)
pred_lagrange  = lagrange_eval(x_data, y_data, x_target)

print(f"\nПРОГНОЗ ПРИ n = {int(x_target)}:")
print("-" * 50)
print(f"{'Многочлен Ньютона':<28} | {pred_newton:>12.6f} $")
print(f"{'Факторіальний многочлен':<28} | {pred_factorial:>12.6f} $")
print(f"{'Многочлен Лагранжа':<28} | {pred_lagrange:>12.6f} $")

plots.plot_main_model(x_data, y_data, newton_coeffs, x_target, pred_newton, OUT_DIR)
plots.plot_tabulation_20n(x_data, y_data, newton_coeffs, OUT_DIR)
plots.plot_nodes_study(x_data, y_data, newton_coeffs, OUT_DIR)

h_fixed = 1000.0
a_fixed = 1000.0
n_sub_configs = [(5, "#d90429"), (10, "#f77f00"), (20, "#2a9d8f")]
plots.plot_fixed_step_study(x_data, newton_coeffs, h_fixed, a_fixed, n_sub_configs, OUT_DIR)

plots.plot_runge_effect(OUT_DIR)
plots.plot_lagrange_comparison(x_data, y_data, newton_coeffs, OUT_DIR)