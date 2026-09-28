import numpy as np
import matplotlib.pyplot as plt
import requests


locations = (
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
    "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
    "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
    "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
)

url = "https://api.open-elevation.com/api/v1/lookup?locations=" + locations

print("Відправляємо запит до API...")
try:
    response = requests.get(url, timeout=15)
    data = response.json()
    points = data["results"]
    print("Успішно отримано точок:", len(points))
except Exception as e:
    print("API недоступне (" + str(e) + "), завантажуємо резервні дані...")
    raw_coords = [tuple(map(float, p.split(','))) for p in locations.split('|')]
    raw_elevs = [
        1264.0, 1285.0, 1285.0, 1333.0, 1310.0, 1318.0, 1318.0, 1339.0,
        1375.0, 1417.0, 1486.0, 1524.0, 1553.0, 1630.0, 1757.0, 1794.0,
        1828.0, 1887.0, 1975.0, 1975.0, 2031.0
    ]
    points = []
    for c, e in zip(raw_coords, raw_elevs):
        points.append({"latitude": c[0], "longitude": c[1], "elevation": e})


f = open("tabulation.txt", "w", encoding="utf-8")
f.write("i\tШирота\tДовгота\tВисота (м)\n")
for i, p in enumerate(points):
    f.write("%d\t%.6f\t%.6f\t%.2f\n" %
            (i, p["latitude"], p["longitude"], p["elevation"]))
f.close()


def get_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

coords = [(p["latitude"], p["longitude"]) for p in points]
elevations = [p["elevation"] for p in points]

distances = [0.0]
for i in range(1, len(points)):
    step = get_distance(coords[i-1][0], coords[i-1][1],
                        coords[i][0], coords[i][1])
    distances.append(distances[-1] + step)


def solve_tridiagonal(alpha, beta, gamma, delta):
    n = len(delta)
    A = [0.0] * n
    B = [0.0] * n
    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]
    for i in range(1, n - 1):
        znam = alpha[i] * A[i-1] + beta[i]
        A[i] = -gamma[i] / znam
        B[i] = (delta[i] - alpha[i] * B[i-1]) / znam
    x = [0.0] * n
    x[-1] = (delta[-1] - alpha[-1] * B[-2]) / (alpha[-1] * A[-2] + beta[-1])
    for i in range(n - 2, -1, -1):
        x[i] = A[i] * x[i+1] + B[i]
    return x, A, B

def build_spline(x, y):
    n = len(x) - 1
    h = [x[i+1] - x[i] for i in range(n)]
    alpha = [0.0] * n
    beta  = [0.0] * n
    gamma = [0.0] * n
    delta = [0.0] * n
    beta[0] = 1.0
    for i in range(1, n - 1):
        alpha[i] = h[i-1]
        beta[i]  = 2.0 * (h[i-1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3.0 * ((y[i+1] - y[i]) / h[i]
                          - (y[i] - y[i-1]) / h[i-1])
    alpha[-1] = h[-2]
    beta[-1]  = 2.0 * (h[-2] + h[-1])
    delta[-1] = 3.0 * ((y[-1] - y[-2]) / h[-1]
                       - (y[-2] - y[-3]) / h[-2])
    c, A_prog, B_prog = solve_tridiagonal(alpha, beta, gamma, delta)
    a = [y[i] for i in range(n)]
    b = [0.0] * n
    d = [0.0] * n
    for i in range(n - 1):
        d[i] = (c[i+1] - c[i]) / (3.0 * h[i])
        b[i] = (y[i+1] - y[i]) / h[i] - (h[i] / 3.0) * (c[i+1] + 2.0 * c[i])
    d[-1] = -c[-1] / (3.0 * h[-1])
    b[-1] = (y[-1] - y[-2]) / h[-1] - (2.0 / 3.0) * h[-1] * c[-1]
    return a, b, c, d, A_prog, B_prog, alpha, beta, gamma, delta

def get_spline_y(x_pts, a, b, c, d, x_val):
    if x_val <= x_pts[0]:
        idx = 0
    elif x_val >= x_pts[-1]:
        idx = len(x_pts) - 2
    else:
        idx = 0
        for i in range(len(x_pts) - 1):
            if x_pts[i] <= x_val <= x_pts[i+1]:
                idx = i
                break
    dx = x_val - x_pts[idx]
    return a[idx] + b[idx] * dx + c[idx] * dx**2 + d[idx] * dx**3

def pick_nodes(x_all, y_all, k):
    idx = np.unique(np.round(np.linspace(0, len(x_all) - 1, k)).astype(int))
    return [x_all[i] for i in idx], [y_all[i] for i in idx]

(a, b, c, d,
 A_prog, B_prog,
 alpha_tbl, beta_tbl, gamma_tbl, delta_tbl) = build_spline(distances, elevations)

max_res = 0.0
for i in range(len(c)):
    left = alpha_tbl[i] * (c[i-1] if i > 0 else 0.0) \
         + beta_tbl[i] * c[i] \
         + gamma_tbl[i] * (c[i+1] if i < len(c) - 1 else 0.0)
    right = delta_tbl[i]
    diff = abs(left - right)
    if diff > max_res:
        max_res = diff
print("\nМаксимальна нев'язка коефіцієнтів с: %.3e" % max_res)

print("\nТАБЛИЦЯ 1: ВУЗЛИ МАРШРУТУ")
print("-" * 40)
print("%4s | %16s | %14s" % ("i", "Відстань x (м)", "Висота y (м)"))
print("-" * 40)
for i in range(len(distances)):
    print("%4d | %16.2f | %14.2f" % (i, distances[i], elevations[i]))


print("\nТАБЛИЦЯ 2: КОЕФІЦІЄНТИ СЛАР")
print("-" * 60)
print("%4s | %11s | %11s | %11s | %11s" %
      ("i", "alpha", "beta", "gamma", "delta"))
print("-" * 60)
for i in range(len(alpha_tbl)):
    print("%4d | %11.4f | %11.4f | %11.4f | %11.4f" %
          (i+1, alpha_tbl[i], beta_tbl[i], gamma_tbl[i], delta_tbl[i]))


print("\nТАБЛИЦЯ 3: ПРОГОНОЧНІ КОЕФІЦІЄНТИ")
print("-" * 42)
print("%4s | %16s | %16s" % ("i", "A_i", "B_i"))
print("-" * 42)
for i in range(len(A_prog)):
    print("%4d | %16.6f | %16.6f" % (i+1, A_prog[i], B_prog[i]))


print("\nТАБЛИЦЯ 4: ЗНАЙДЕНІ КОЕФІЦІЄНТИ c")
print("-" * 50)
for i in range(len(c)):
    print("c[%2d] = %14.6e" % (i, c[i]))


print("\nТАБЛИЦЯ 5: КОЕФІЦІЄНТИ СПЛАЙНІВ")
print("-" * 70)
print("%4s | %10s | %12s | %14s | %16s" % ("i", "a", "b", "c", "d"))
print("-" * 70)
for i in range(len(a)):
    print("%4d | %10.2f | %12.5f | %14.7f | %16.9f" %
          (i, a[i], b[i], c[i], d[i]))


x_fine = np.linspace(distances[0], distances[-1], 500)
y_full = [get_spline_y(distances, a, b, c, d, val) for val in x_fine]

plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 0.8


fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
ax.plot(distances, elevations, linestyle='-', color='#1d3557', linewidth=1.5, alpha=0.7)
ax.scatter(distances, elevations, color='#e63946', edgecolor='white',
           linewidth=1.0, s=45, zorder=5, label="Опорні вузли маршруту")

ax.set_title("Дискретний графік висот: База «Заросляк» — г. Говерла",
             fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel("Кумулятивна відстань, м", fontsize=10)
ax.set_ylabel("Висота над рівнем моря, м", fontsize=10)
ax.set_xlim(0, distances[-1])

ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7, color='#b0bec5')
ax.minorticks_on()
ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.4, color='#cfd8dc')
ax.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='#b0bec5')

plt.tight_layout()
plt.savefig("profile_discrete.png", dpi=300)
plt.close()


fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
ax.fill_between(x_fine, y_full, min(elevations) - 25,
                color='#a8dadc', alpha=0.35, label="Профіль рельєфу")
ax.plot(x_fine, y_full, color='#1d3557', linewidth=2.0,
        label="Кубічний сплайн (21 вузол)")
ax.scatter(distances, elevations, color='#e63946', edgecolor='white',
           linewidth=1.0, s=40, zorder=5, label="Вихідні точки інтерполяції")

ax.set_title("Побудова гладкого профілю маршруту кубічним сплайном",
             fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel("Кумулятивна відстань, м", fontsize=10)
ax.set_ylabel("Висота над рівнем моря, м", fontsize=10)
ax.set_xlim(0, distances[-1])
ax.set_ylim(min(elevations) - 25, max(elevations) + 25)

ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7, color='#b0bec5')
ax.minorticks_on()
ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.4, color='#cfd8dc')
ax.legend(loc="upper left", frameon=True, facecolor='white',
          framealpha=0.9, edgecolor='#b0bec5')

plt.tight_layout()
plt.savefig("spline_21_nodes.png", dpi=300)
plt.close()


schemes = [
    (10, '#d90429', "Сплайн на 10 вузлах"),
    (15, '#f77f00', "Сплайн на 15 вузлах"),
    (20, '#2a9d8f', "Сплайн на 20 вузлах")
]

spline_curves = {}
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300, sharey=True)

for idx, (k, clr, lbl) in enumerate(schemes):
    ax = axes[idx]
    xk, yk = pick_nodes(distances, elevations, k)
    ak, bk, ck, dk, _, _, _, _, _, _ = build_spline(xk, yk)
    yk_fine = [get_spline_y(xk, ak, bk, ck, dk, val) for val in x_fine]
    spline_curves[k] = yk_fine

    ax.plot(x_fine, y_full, color='#b0bec5', linestyle='--', linewidth=1.4,
            label="Еталон (21 вузол)")
    ax.plot(x_fine, yk_fine, color=clr, linewidth=2.0, label=lbl)
    ax.scatter(xk, yk, color='#1d3557', edgecolor='white', linewidth=1.0,
               s=55, zorder=5, label=f"Опорні вузли ({k})")

    ax.set_title(f"Інтерполяція на {k} вузлах",
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel("Кумулятивна відстань, м", fontsize=10)
    if idx == 0:
        ax.set_ylabel("Висота над рівнем моря, м", fontsize=10)

    ax.set_xlim(0, distances[-1])
    ax.set_ylim(min(elevations) - 25, max(elevations) + 25)

    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7, color='#b0bec5')
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.4, color='#cfd8dc')
    ax.legend(loc="upper left", fontsize=9, frameon=True, facecolor='white',
              framealpha=0.9, edgecolor='#b0bec5')

fig.suptitle("Порівняння кубічних сплайнів для різної кількості вузлів",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("comparison_nodes.png", dpi=300, bbox_inches='tight')
plt.close()


errors = {}
for k, clr, lbl in schemes:
    yk_fine = spline_curves[k]
    err = [abs(y_full[i] - yk_fine[i]) for i in range(len(x_fine))]
    errors[k] = err

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
for k, clr, lbl in schemes:
    ax.plot(x_fine, errors[k], color=clr, linewidth=1.8,
            label=f"Похибка ε для {k} вузлів")

ax.set_title("Графік абсолютної похибки наближення ε = |f(x) - y_набл|",
             fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel("Кумулятивна відстань, м", fontsize=10)
ax.set_ylabel("Абсолютна похибка, м", fontsize=10)

ax.set_xlim(left=0, right=distances[-1])
max_err_val = max(max(errors[k]) for k in (10, 15, 20))
ax.set_ylim(bottom=0, top=max_err_val * 1.05)

ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7, color='#b0bec5')
ax.minorticks_on()
ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.4, color='#cfd8dc')
ax.legend(loc="upper right", frameon=True, facecolor='white',
          framealpha=0.9, edgecolor='#b0bec5')

plt.tight_layout()
plt.savefig("error_plot.png", dpi=300)
plt.close()


n_pts = len(elevations)
total_ascent  = sum(max(elevations[i] - elevations[i-1], 0) for i in range(1, n_pts))
total_descent = sum(max(elevations[i-1] - elevations[i], 0) for i in range(1, n_pts))

grad = np.gradient(y_full, x_fine) * 100.0

mass = 80.0
g = 9.81
energy_j = mass * g * total_ascent


print("\nРЕЗУЛЬТАТИ ОЦІНКИ ТОЧНОСТІ")
print("-" * 44)
print("%10s | %14s | %14s" % ("Вузли", "max похибка, м", "mean похибка, м"))
print("-" * 44)
for k in (10, 15, 20):
    e = errors[k]
    print("%10d | %14.3f | %14.3f" % (k, max(e), sum(e) / len(e)))


print("\nХАРАКТЕРИСТИКИ МАРШРУТУ ТА ЕНЕРГОВИТРАТИ")
print("-" * 60)
print("Довжина маршруту:           %.2f м" % distances[-1])
print("Сумарний підйом:            %.2f м" % total_ascent)
print("Сумарний спуск:             %.2f м" % total_descent)
print("Максимальний підйом:        %.2f %%" % np.max(grad))
print("Максимальний спуск:         %.2f %%" % np.min(grad))
print("Середній ухил:              %.2f %%" % np.mean(np.abs(grad)))
print("Кількість точок |grad| > 15%%: %d з %d" %
      (int(np.sum(np.abs(grad) > 15)), len(grad)))
print("Механічна робота:           %.2f Дж (%.2f кДж)" %
      (energy_j, energy_j / 1000))
print("Витрачена енергія:          %.2f ккал" % (energy_j / 4184))