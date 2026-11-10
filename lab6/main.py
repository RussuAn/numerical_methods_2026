from pathlib import Path

from methods import (
    read_matrix,
    read_vector,
    write_lu,
    write_vector,
    lu_decomposition,
    solve_lu,
    residual,
    vector_norm,
    refine_solution,
)


def main():
    folder = Path(__file__).resolve().parent / "data"
    eps = 1e-14

    A = read_matrix(folder / "A.txt")
    b = read_vector(folder / "B.txt")

    L, U = lu_decomposition(A)
    write_lu(folder / "LU.txt", L, U)

    x_initial = solve_lu(L, U, b)
    initial_residual = vector_norm(
        residual(A, x_initial, b)
    )

    write_vector(folder / "X_initial.txt", x_initial)

    x, history, converged = refine_solution(
        A, b, L, U, x_initial, eps
    )

    write_vector(folder / "X_refined.txt", x)

    lines = [
        f"Розмірність системи: {len(b)}",
        f"Задана точність: {eps:.6e}",
        f"Норма нев'язки до уточнення: {initial_residual:.6e}",
        "",
        f"{'Ітерація':>10} {'Норма поправки':>20} {'Норма нев’язки':>20}",
    ]

    for iteration, correction_norm, residual_norm in history:
        lines.append(
            f"{iteration:>10} "
            f"{correction_norm:>20.6e} "
            f"{residual_norm:>20.6e}"
        )

    lines.extend([
        "",
        f"Кількість ітерацій: {len(history)}",
        f"Задану точність досягнуто: {'так' if converged else 'ні'}",
        "",
        f"{'i':>5} {'Початкове x':>22} {'Уточнене x':>22}",
    ])

    for i in range(len(x)):
        lines.append(
            f"{i + 1:>5} "
            f"{x_initial[i]:>22.16f} "
            f"{x[i]:>22.16f}"
        )

    result = "\n".join(lines)
    print(result)

    (folder / "results.txt").write_text(
        result + "\n",
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()