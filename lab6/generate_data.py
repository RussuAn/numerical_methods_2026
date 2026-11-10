import random
from pathlib import Path

from methods import matrix_vector, write_matrix, write_vector


def main():
    n = 100
    random.seed(42)

    A = [
        [random.randint(-10, 10) for _ in range(n)]
        for _ in range(n)
    ]

    x = [2.5] * n
    b = matrix_vector(A, x)

    folder = Path(__file__).resolve().parent

    write_matrix(folder / "A.txt", A)
    write_vector(folder / "B.txt", b)

    print("Збережено A.txt (100 × 100) та B.txt (100 елементів).")


if __name__ == "__main__":
    main()