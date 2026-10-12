import csv
import numpy as np


def init_csv_file(filepath):
    rows = [
        (1, -2), (2, 0), (3, 5), (4, 10), (5, 15), (6, 20),
        (7, 23), (8, 22), (9, 17), (10, 10), (11, 5), (12, 0),
        (13, -10), (14, 3), (15, 7), (16, 13), (17, 19), (18, 20),
        (19, 22), (20, 21), (21, 18), (22, 15), (23, 10), (24, 3)
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Month", "Temp"])
        for r in rows:
            w.writerow(r)


def read_data(filename):
    x, y = [], []
    with open(filename, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row["Month"]))
            y.append(float(row["Temp"]))
    return np.array(x, dtype=float), np.array(y, dtype=float)