import csv
from pathlib import Path
import matplotlib.pyplot as plt
from functions import f
from tabulation import tabulate, DATA, PLOTS
from iterative_methods import run_all_for_interval
from polynomial_methods import save_coefficients, read_coefficients, horner, newton_horner, lin

def save_csv(filename, headers, records):
    with open(DATA / filename, 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(records)

def draw_methods(all_results):
    names = list(all_results[0].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(len(names)))
    width = 0.37
    for index, results in enumerate(all_results):
        iterations = [results[name][1] for name in names]
        ax.bar([x+(index-.5)*width for x in xs], iterations, width,
               label=f'Корінь {index+1}')
    ax.set_xticks(xs, names, rotation=20, ha='right')
    ax.set_ylabel('Кількість ітерацій')
    ax.set_title('Порівняння шести методів для двох коренів')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / 'iterations.png', dpi=180)
    plt.close(fig)

    for index, results in enumerate(all_results, 1):
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, (_, _, history) in results.items():
            ax.semilogy([t[0] for t in history], [max(t[2], 1e-17) for t in history],
                        marker='o', label=name)
        ax.set(title=f'Збіжність методів: корінь {index}', xlabel='Ітерація',
               ylabel='|F(x)|')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS / f'convergence_root_{index}.png', dpi=180)
        plt.close(fig)

def draw_polynomial(coefficients, roots):
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [-1 + i * 0.02 for i in range(251)]
    ax.plot(xs, [horner(coefficients, x) for x in xs])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(roots, linestyle='--', label=f'Дійсний корінь: {roots:.10f}')
    ax.set(title='Кубічне рівняння', xlabel='x', ylabel='P(x)')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / 'polynomial.png', dpi=180)
    plt.close(fig)

def main():
    _, intervals = tabulate()
    if len(intervals) != 2:
        raise RuntimeError(f'Потрібні рівно два інтервали зміни знака, отримано: {intervals}')
    results = []
    rows = []
    print('ФУНКЦІЯ: F(x) = sin(x) - 0.3*x + 0.1')
    print('Діапазон [-1; 4], крок h=0.1, точність eps=1e-10')
    for index, interval in enumerate(intervals, 1):
        print(f'\nКорінь {index}; інтервал {interval}:')
        current = run_all_for_interval(interval)
        results.append(current)
        for name, (x, count, history) in current.items():
            residue = abs(f(x))
            diff = history[-1][3]
            rows.append([index, interval[0], interval[1], name,
                         f'{x:.15f}', count, f'{residue:.3e}', f'{diff:.3e}'])
            print(f'{name:23s} x={x:.15f}, ітерацій={count:3d}, '
                  f'|F(x)|={residue:.3e}, |dx|={diff:.3e}')
    save_csv('methods_results.csv',
             ['Корінь', 'Ліва межа', 'Права межа', 'Метод', 'Корінь x',
              'Ітерацій', '|F(x)|', '|x(k+1)-x(k)|'], rows)
    draw_methods(results)

    save_coefficients()
    coefficients = read_coefficients()
    real_root, real_iters, hist_newton = newton_horner(coefficients)
    complex_roots, complex_iters, hist_lin = lin(coefficients)
    print('\nКУБІЧНЕ РІВНЯННЯ: x^3 - 4x^2 + 6x - 4 = 0')
    print(f'Ньютон + Горнер: x={real_root:.15f}, ітерацій={real_iters}, '
          f'|P(x)|={abs(horner(coefficients, real_root)):.3e}')
    print(f'Метод Ліна: z1={complex_roots[0]}, z2={complex_roots[1]}, '
          f'ітерацій={complex_iters}')
    for root in complex_roots:
        print(f'  |P({root})|={abs(horner(coefficients, root)):.3e}')
    save_csv('polynomial_results.csv', ['Метод', 'Корінь', 'Ітерацій', '|P(x)|'],
             [['Ньютон-Горнер', real_root, real_iters, abs(horner(coefficients, real_root))]] +
             [['Ліна', z, complex_iters, abs(horner(coefficients, z))] for z in complex_roots])
    draw_polynomial(coefficients, real_root)
    print('\nРезультати збережено в data/, графіки — у plots/.')

if __name__ == '__main__':
    main()