import csv
import numpy as np

def init_csv_file(filepath):
    rows = [
        (1000,  0.2),
        (2000,  0.45),
        (5000,  1.5),
        (10000, 3.2),
        (20000, 7.5),
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "t"])
        for r in rows:
            w.writerow(r)

def read_data(filename):
    x, y = [], []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return np.array(x, dtype=float), np.array(y, dtype=float)