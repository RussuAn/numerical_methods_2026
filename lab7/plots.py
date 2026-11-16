from pathlib import Path
import matplotlib.pyplot as plt


def save(name, out_dir):
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / name, dpi=200)
    plt.close()


def plot_convergence(h_simple, h_jacobi, h_seidel, out_dir):
    plt.figure(figsize=(8, 5))

    plt.semilogy(
        range(1, len(h_simple) + 1),
        h_simple, label="Проста ітерація"
    )
    plt.semilogy(
        range(1, len(h_jacobi) + 1),
        h_jacobi, label="Метод Якобі"
    )
    plt.semilogy(
        range(1, len(h_seidel) + 1),
        h_seidel, label="Метод Зейделя"
    )

    plt.xlabel("Номер ітерації")
    plt.ylabel("Норма нев'язки")
    plt.title("Збіжність ітераційних методів")
    plt.legend()

    save("convergence.png", out_dir)


def plot_iterations(counts, out_dir):
    methods = ["Проста ітерація", "Якобі", "Зейделя"]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, counts)

    plt.xlabel("Метод")
    plt.ylabel("Кількість ітерацій")
    plt.title("Порівняння кількості ітерацій")

    save("iterations.png", out_dir)


def plot_errors(errors, out_dir):
    methods = ["Проста ітерація", "Якобі", "Зейделя"]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, errors)
    plt.yscale("log")

    plt.xlabel("Метод")
    plt.ylabel("Абсолютна похибка")
    plt.title("Порівняння похибок методів")

    save("errors.png", out_dir)