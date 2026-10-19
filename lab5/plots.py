import os
import numpy as np
import matplotlib.pyplot as plt

def save(name, out_dir):
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, name), dpi=200)
    plt.close()

def plot_function(f, a, b, out_dir):
    x = np.linspace(a, b, 1000)
    plt.figure(figsize=(8, 4))
    plt.plot(x, f(x))
    plt.xlabel("Час x, год")
    plt.ylabel("Навантаження f(x)")
    plt.title("Навантаження на сервер")
    save("function.png", out_dir)

def plot_errors(ns, errors, n_opt, epsilon, out_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(ns, errors)
    plt.yscale("symlog", linthresh=epsilon / 10)
    plt.axhline(
        epsilon, color="red", linestyle="--",
        label="Задана точність"
    )
    index = np.where(ns == n_opt)[0][0]
    plt.scatter(
        n_opt, errors[index],
        color="red", label=f"N_opt = {n_opt}"
    )
    plt.xlabel("Число розбиттів N")
    plt.ylabel("Абсолютна похибка")
    plt.title("Похибка складової формули Сімпсона")
    plt.legend()
    save("simpson_error.png", out_dir)

def plot_comparison(names, errors, out_dir):
    plt.figure(figsize=(8, 4))
    plt.bar(names, errors)
    plt.yscale("symlog", linthresh=1e-13)
    plt.ylabel("Абсолютна похибка")
    plt.title("Порівняння методів")
    save("comparison.png", out_dir)


def plot_adaptive(deltas, errors, counts, out_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(deltas, errors, "o-")
    plt.xscale("log")
    plt.yscale("symlog", linthresh=1e-13)
    plt.xlabel("Параметр δ")
    plt.ylabel("Абсолютна похибка")
    plt.title("Точність адаптивного алгоритму")
    save("adaptive_error.png", out_dir)
    plt.figure(figsize=(8, 4))
    plt.semilogx(deltas, counts, "o-")
    plt.xlabel("Параметр δ")
    plt.ylabel("Кількість обчислень f(x)")
    plt.title("Обчислювальні витрати адаптивного алгоритму")
    save("adaptive_count.png", out_dir)