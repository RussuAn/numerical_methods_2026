from pathlib import Path
import matplotlib.pyplot as plt
from functions import f

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
PLOTS = BASE / 'plots'

def tabulate(a=-1.0, b=4.0, h=0.1):
    DATA.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    n = round((b - a) / h)
    points = [(round(a + i * h, 10), f(round(a + i * h, 10))) for i in range(n + 1)]
    with open(DATA / 'tabulation.txt', 'w', encoding='utf-8') as file:
        file.write('x\tF(x)\n')
        for x, y in points:
            file.write(f'{x:.10f}\t{y:.15f}\n')

    intervals = []
    for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
        if y1 * y2 < 0:
            intervals.append((x1, x2))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([x for x, _ in points], [y for _, y in points], marker='.', label='F(x)')
    ax.axhline(0, color='black', linewidth=0.8)
    for i, (left, right) in enumerate(intervals, 1):
        ax.axvspan(left, right, alpha=0.18, label=f'Корінь {i}: [{left:.1f}; {right:.1f}]')
    ax.set(title='Графік трансцендентної функції', xlabel='x', ylabel='F(x)')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / 'function.png', dpi=180)
    plt.close(fig)
    return points, intervals

if __name__ == '__main__':
    points, intervals = tabulate()
    print(f'Табульовано точок: {len(points)}')
    print(f'Інтервали зміни знака: {intervals}')