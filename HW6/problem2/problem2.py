import os
import numpy as np
import matplotlib.pyplot as plt



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# stability functions
def R_euler(z):
    return 1.0 + z


def R_rk2(z):
    return 1.0 + z + 0.5 * z**2


def R_rk4(z):
    return 1.0 + z + 0.5 * z**2 + (1.0 / 6.0) * z**3 + (1.0 / 24.0) * z**4


# estimate x_min
def estimate_xmin(R_func, x_left=-5.0, x_right=0.0, num_points=200000):
    x = np.linspace(x_left, x_right, num_points)
    modR = np.abs(R_func(x))
    stable = modR <= 1.0

    if not np.any(stable):
        return None

    return np.min(x[stable])


# plotting contour of stability regions
def plot_stability_regions(methods, filename=None):
    if filename is None:
        filename = os.path.join(SCRIPT_DIR, "problem2_stability_regions.png")

    x = np.linspace(-5.0, 5.0, 1000)
    y = np.linspace(-5.0, 5.0, 1000)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    colors = {
        "Euler": "blue",
        "RK2": "orange",
        "RK4": "green"
    }

    plt.figure(figsize=(8, 8))

    for name, R_func in methods.items():
        modR = np.abs(R_func(Z))
        plt.contour(X, Y, modR, levels=[1.0], colors=[colors[name]], linewidths=2)
        plt.plot([], [], color=colors[name], label=name)

    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axvline(0.0, color="black", linewidth=0.8)

    plt.xlabel(r"$\mathrm{Re}(z)$")
    plt.ylabel(r"$\mathrm{Im}(z)$")
    plt.title("Problem 2: Stability Regions of Explicit Methods")

    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(filename, dpi=200)
    plt.show()

    print(f"Saved stability region plot to: {filename}")


# x minimum output
def save_xmin_summary(methods, filename=None):
    if filename is None:
        filename = os.path.join(SCRIPT_DIR, "problem2_summary.txt")

    print("\nEstimated leftmost stable point on negative real axis")
    print("-" * 60)
    print(f"{'Method':<10s} {'x_min':>15s}")

    with open(filename, "w", encoding="utf-8") as file:
        file.write("Method x_min Notes\n")

        for name, R_func in methods.items():
            xmin = estimate_xmin(R_func)
            print(f"{name:<10s} {xmin:15.8f}")
            file.write(f"{name} {xmin:.10f} estimated_from_real_axis_scan\n")

    print(f"\nSaved x_min summary to: {filename}")

if __name__ == "__main__":
    print("Problem 2: Stability regions of explicit methods")
    print("=" * 55)
    print(f"Script directory: {SCRIPT_DIR}")

    methods = {
        "Euler": R_euler,
        "RK2": R_rk2,
        "RK4": R_rk4
    }

    plot_stability_regions(methods)
    save_xmin_summary(methods)