import os
import numpy as np
import matplotlib.pyplot as plt
from interpolation import newton_eval, omega_k, divided_differences, lagrange_eval, runge_function

plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8

def plot_main_model(x_data, y_data, newton_coeffs, x_target, pred_val, out_dir):
    xf = np.linspace(x_data[0], x_data[-1], 300)
    yf = [newton_eval(x_data, newton_coeffs, v) for v in xf]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(xf, yf, color='#1d3557', linewidth=2.0, label="Многочлен Ньютона $N_4(x)$")
    ax.scatter(x_data, y_data, color='#e63946', s=60, zorder=5, label="Експериментальні точки")
    ax.scatter([x_target], [pred_val], marker='*', s=220, color='gold', edgecolor='black', zorder=6,
               label=f"Прогноз при n={int(x_target)}: {pred_val:.3f} $")
    ax.axvline(x_target, color='gray', linestyle=':', linewidth=1)
    ax.set_title("Модель Cost = f(Tasks) за многочленом Ньютона", fontweight='bold', pad=10)
    ax.set_xlabel("Tasks (n)")
    ax.set_ylabel("Cost ($)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_main.png"), dpi=300)
    plt.close()

def plot_tabulation_20n(x_data, y_data, newton_coeffs, out_dir):
    n_pts = len(x_data)
    h_step = (x_data[-1] - x_data[0]) / (20.0 * n_pts)
    x_tab = np.arange(x_data[0], x_data[-1] + h_step / 2.0, h_step)

    n_tab_vals = [newton_eval(x_data, newton_coeffs, v) for v in x_tab]
    omega_vals = [omega_k(v, x_data, n_pts - 1) for v in x_tab]
    c_n = abs(newton_coeffs[-1])
    err_est = [c_n * abs(w) for w in omega_vals]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=300)
    
    axes[0].plot(x_tab, n_tab_vals, color='#1d3557', linewidth=1.8, label="$N_4(x)$")
    axes[0].scatter(x_data, y_data, color='#e63946', s=40, zorder=5, label="Вузли")
    axes[0].set_title("Многочлен Ньютона $N_4(x)$", fontweight='bold')
    axes[0].set_xlabel("Tasks (n)")
    axes[0].set_ylabel("Cost ($)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    axes[1].plot(x_tab, err_est, color='#d90429', linewidth=1.8, label="Оцінка $R_4(x)$")
    axes[1].set_title("Теоретичний залишок $R_4(x)$", fontweight='bold')
    axes[1].set_xlabel("Tasks (n)")
    axes[1].set_ylabel("Похибка ($)")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    axes[2].plot(x_tab, omega_vals, color='#7209b7', linewidth=1.8, label="$\omega_4(x)$")
    axes[2].set_title("Вузлова функція $\omega_4(x)$", fontweight='bold')
    axes[2].set_xlabel("Tasks (n)")
    axes[2].set_ylabel("$\omega_4(x)$")
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_tabulation_20n.png"), dpi=300)
    plt.close()

def plot_nodes_study(x_data, y_data, full_coeffs, out_dir):
    xf = np.linspace(x_data[0], x_data[-1], 300)
    base_curve = [newton_eval(x_data, full_coeffs, v) for v in xf]

    schemes = [
        (5,  '#d90429', "5 вузлів"),
        (10, '#f77f00', "10 вузлів"),
        (20, '#2a9d8f', "20 вузлів"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300, sharey=True)
    fig_err, ax_err = plt.subplots(figsize=(8, 4.5), dpi=300)

    for idx, (k, clr, lbl) in enumerate(schemes):
        xk = np.linspace(x_data[0], x_data[-1], k)
        yk = [newton_eval(x_data, full_coeffs, v) for v in xk]
        ck = list(divided_differences(xk, yk)[0, :])

        yk_fine = [newton_eval(xk, ck, v) for v in xf]
        err_k = [abs(base_curve[i] - yk_fine[i]) for i in range(len(xf))]

        axes[idx].plot(xf, base_curve, color='#b0bec5', linestyle='--', linewidth=1.5, label="Базова модель")
        axes[idx].plot(xf, yk_fine, color=clr, linewidth=1.8, label=lbl)
        axes[idx].scatter(xk, yk, color='#1d3557', s=40, zorder=5)
        axes[idx].set_title(f"Інтерполяція на {k} вузлах", fontweight='bold')
        axes[idx].set_xlabel("Tasks (n)")
        if idx == 0:
            axes[idx].set_ylabel("Cost ($)")
        axes[idx].grid(True, linestyle='--', alpha=0.6)
        axes[idx].legend(loc="upper left")

        ax_err.plot(xf, err_k, color=clr, linewidth=1.8, label=f"Похибка для {k} вузлів")

    fig.suptitle("Порівняння многочленів Ньютона для різної кількості вузлів", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_nodes.png"), dpi=300)
    plt.close(fig)

    ax_err.set_title("Графік абсолютної похибки наближення для різної кількості вузлів", fontweight='bold', pad=10)
    ax_err.set_xlabel("Tasks (n)")
    ax_err.set_ylabel("Абсолютна похибка ($)")
    ax_err.grid(True, linestyle='--', alpha=0.6)
    ax_err.legend(loc="upper right")
    fig_err.tight_layout()
    fig_err.savefig(os.path.join(out_dir, "fig_errors.png"), dpi=300)
    plt.close(fig_err)

def plot_fixed_step_study(x_data, full_coeffs, h_fixed, a_fixed, n_configs, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    
    for idx, (n_val, clr) in enumerate(n_configs):
        b_val = a_fixed + h_fixed * n_val
        xs = np.array([a_fixed + i * h_fixed for i in range(n_val + 1)])
        ys = [newton_eval(x_data, full_coeffs, v) for v in xs]
        ck = list(divided_differences(xs, ys)[0, :])

        xf_local = np.linspace(a_fixed, b_val, 250)
        yf_local = [newton_eval(xs, ck, v) for v in xf_local]
        y_ref_local = [newton_eval(x_data, full_coeffs, v) for v in xf_local]

        axes[idx].plot(xf_local, y_ref_local, color='#b0bec5', linestyle='--', label="Базова модель")
        axes[idx].plot(xf_local, yf_local, color=clr, linewidth=1.8, label=f"Ньютон ({len(xs)} вузлів)")
        axes[idx].scatter(xs, ys, color='#1d3557', s=35, zorder=5)
        axes[idx].set_title(f"n={n_val}: [{int(a_fixed)}; {int(b_val)}], h={int(h_fixed)}", fontweight='bold')
        axes[idx].set_xlabel("Tasks (n)")
        if idx == 0:
            axes[idx].set_ylabel("Cost ($)")
        axes[idx].grid(True, linestyle='--', alpha=0.6)
        axes[idx].legend(loc="upper left")

    fig.suptitle(f"Дослідження при фіксованому кроці h = {int(h_fixed)}", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_fixed_h.png"), dpi=300)
    plt.close()

def plot_runge_effect(out_dir):
    xr = np.linspace(-1, 1, 400)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(xr, runge_function(xr), color='#1d3557', linewidth=2.2, label="f(x)")

    for k, clr in zip((5, 11, 20), ["#d90429", "#f77f00", "#2a9d8f"]):
        xk = np.linspace(-1, 1, k)
        yk = runge_function(xk)
        ck = list(divided_differences(xk, yk)[0, :])
        ax.plot(xr, [newton_eval(xk, ck, v) for v in xr], color=clr, linewidth=1.5, label=f"Ньютон ({k} вузлів)")

    ax.set_ylim(-0.4, 1.4)
    ax.set_title("Демонстрація ефекту Рунге", fontweight='bold', pad=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_runge.png"), dpi=300)
    plt.close()

def plot_lagrange_comparison(x_data, y_data, newton_coeffs, out_dir):
    xf = np.linspace(x_data[0], x_data[-1], 300)
    y_newton = [newton_eval(x_data, newton_coeffs, v) for v in xf]
    y_lagrange = [lagrange_eval(x_data, y_data, v) for v in xf]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(xf, y_newton, color='#1d3557', linewidth=2.0, label="Форма Ньютона")
    ax.plot(xf, y_lagrange, color='#e63946', linestyle='--', linewidth=1.8, label="Форма Лагранжа")
    ax.scatter(x_data, y_data, color='gold', edgecolor='black', s=60, zorder=5, label="Вхідні вузли")
    ax.set_title("Порівняння інтерполяції за Ньютоном та Лагранжем", fontweight='bold', pad=10)
    ax.set_xlabel("Tasks (n)")
    ax.set_ylabel("Cost ($)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_lagrange.png"), dpi=300)
    plt.close()