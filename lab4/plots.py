from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from methods import moisture, exact_derivative

def plot_function(output_dir, t_start=0, t_end=20):
    t = np.linspace(t_start, t_end, 500)
    plt.figure(figsize=(9, 5))
    plt.plot(t, moisture(t), label='M(t)')
    plt.plot(t, exact_derivative(t), label="M'(t)")
    plt.xlabel('t')
    plt.ylabel('Значення')
    plt.title('Функція вологості та її похідна')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'function.png', dpi=160)
    plt.close()

def plot_step_error(steps, errors, best_h, best_error, output_dir):
    positive = errors > 0
    plt.figure(figsize=(9, 5))
    plt.loglog(steps[positive], errors[positive], label='Абсолютна похибка')
    if best_error > 0:
        plt.scatter(best_h, best_error, label='Найкращий крок на сітці')
    plt.xlabel('Крок h')
    plt.ylabel('Абсолютна похибка')
    plt.title('Залежність похибки від кроку')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'error.png', dpi=160)
    plt.close()

def plot_comparison(values, exact, output_dir):
    labels = ['Центральна\nрізниця', 'Рунге–\nРомберг', 'Ейткен']
    errors = [abs(value - exact) for value in values]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, errors)
    plt.yscale('log')
    plt.ylabel('Абсолютна похибка')
    plt.title('Порівняння точності методів')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'comparison.png', dpi=160)
    plt.close()