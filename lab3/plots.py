import os
import numpy as np
import matplotlib.pyplot as plt
from least_squares import eval_polynomial, gauss_solve, form_matrix, form_vector

plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8


def plot_variance_vs_degree(degrees, variances, optimal_m, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(degrees, variances, marker='o', color='#1d3557', linewidth=2.0, label=r'Дисперсія $\delta(m)$')
    ax.axvline(optimal_m, color='#e63946', linestyle='--', linewidth=1.5, label=f'Оптимальний m = {optimal_m}')
    ax.scatter([optimal_m], [variances[optimal_m - 1]], color='gold', edgecolor='black', s=120, zorder=6)
    ax.set_title("Залежність дисперсії від степеня полінома m", fontweight='bold', pad=10)
    ax.set_xlabel("Степінь многочлена m")
    ax.set_ylabel("Дисперсія delta (°C)")
    ax.set_xticks(degrees)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_variance_vs_degree.png"), dpi=300)
    plt.close()


def plot_approx_and_forecast(x, y, coef, optimal_m, x_future, y_future, out_dir):
    xf = np.linspace(x[0], x[-1], 300)
    yf = eval_polynomial(xf, coef)
    xf_proj = np.linspace(x[-1], x_future[-1], 100)
    yf_proj = eval_polynomial(xf_proj, coef)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(xf, yf, color='#1d3557', linewidth=2.0, label=f"МНК-апроксимація (m={optimal_m})")
    ax.plot(xf_proj, yf_proj, color='#e63946', linestyle='--', linewidth=1.8, label="Екстраполяція тренду")
    ax.scatter(x, y, color='#457b9d', s=45, zorder=5, label="Фактичні спостереження")
    ax.scatter(x_future, y_future, marker='*', s=180, color='gold', edgecolor='black', zorder=6,
               label=f"Прогноз на міс. 25-27")

    for xf_val, yf_val in zip(x_future, y_future):
        ax.annotate(f"{yf_val:.1f}°C", (xf_val, yf_val), textcoords="offset points", xytext=(0, 10), ha='center',
                    fontsize=8, fontweight='bold')

    ax.set_title("Апроксимація температури та прогноз на 3 місяці", fontweight='bold', pad=10)
    ax.set_xlabel("Місяць")
    ax.set_ylabel("Температура (°C)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_approx_and_forecast.png"), dpi=300)
    plt.close()


def plot_error_tabulation(x, y, all_coefs, optimal_m, out_dir):
    n_pts = len(x)
    h1 = (x[-1] - x[0]) / (20.0 * n_pts)
    x_tab = np.arange(x[0], x[-1] + h1 / 2.0, h1)
    y_true_interp = np.interp(x_tab, x, y)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    selected_degrees = [(2, '#a8dadc'), (4, '#457b9d'), (optimal_m, '#1d3557'), (10, '#e63946')]

    for deg, color in selected_degrees:
        y_app_tab = eval_polynomial(x_tab, all_coefs[deg])
        err_tab = np.abs(y_true_interp - y_app_tab)
        ax.plot(x_tab, err_tab, color=color, linewidth=1.6, label=f"m = {deg}")

    ax.set_title("Табуляція абсолютної похибки апроксимації з кроком h1", fontweight='bold', pad=10)
    ax.set_xlabel("Місяць")
    ax.set_ylabel("Похибка |f(x) - phi(x)| (°C)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="upper right", frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_error_tabulation.png"), dpi=300)
    plt.close()