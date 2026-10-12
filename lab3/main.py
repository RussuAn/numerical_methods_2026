import os
import numpy as np
from data_processing import init_csv_file, read_data
from least_squares import form_matrix, form_vector, gauss_solve, eval_polynomial, calc_variance
import plots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data.csv")
OUT_DIR = os.path.join(BASE_DIR, "output_plots")
os.makedirs(OUT_DIR, exist_ok=True)

init_csv_file(CSV_PATH)
x_data, y_data = read_data(CSV_PATH)

print("-" * 35)
print(f"{'Місяць (x)':>12} | {'Температура (y)':>16}")
print("-" * 35)
for xi, yi in zip(x_data, y_data):
    print(f"{xi:>12.0f} | {yi:>16.1f}")

degrees = list(range(1, 11))
variances = []
all_coefs = {}

print("\n" + "-" * 55)
print(f"{'Степінь m':>10} | {'Дисперсія delta (°C)':>22} | {'Макс. похибка':>16}")
print("-" * 55)

for m in degrees:
    A_mat = form_matrix(x_data, m)
    b_vec = form_vector(x_data, y_data, m)
    coef = gauss_solve(A_mat, b_vec)
    all_coefs[m] = coef
    y_app = eval_polynomial(x_data, coef)
    var = calc_variance(y_data, y_app)
    variances.append(var)
    max_err = np.max(np.abs(y_data - y_app))
    print(f"{m:>10} | {var:>22.6f} | {max_err:>16.6f}")

optimal_m = 6
best_coef = all_coefs[optimal_m]

print("\n" + "-" * 40)
print(f"КОЕФІЦІЄНТИ ПОЛІНОМА (m = {optimal_m}):")
print("-" * 40)
for i, c in enumerate(best_coef):
    print(f"a[{i}] = {c:>16.8e}")

x_future = np.array([25.0, 26.0, 27.0])
y_future = eval_polynomial(x_future, best_coef)

print("\n" + "-" * 40)
print("ПРОГНОЗ НА НАСТУПНІ 3 МІСЯЦІ:")
print("-" * 40)
for xf, yf in zip(x_future, y_future):
    print(f"Місяць {int(xf)}: {yf:>10.2f} °C")

plots.plot_variance_vs_degree(degrees, variances, optimal_m, OUT_DIR)
plots.plot_approx_and_forecast(x_data, y_data, best_coef, optimal_m, x_future, y_future, OUT_DIR)
plots.plot_error_tabulation(x_data, y_data, all_coefs, optimal_m, OUT_DIR)