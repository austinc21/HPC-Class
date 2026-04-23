import os
import time
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import cupy as cp
    try:
        _test = cp.arange(1)
        cp.cuda.Stream.null.synchronize()
        GPU_AVAILABLE = True
    except Exception:
        cp = None
        GPU_AVAILABLE = False
except ImportError:
    cp = None
    GPU_AVAILABLE = False

def get_xp(use_gpu=False):
    if use_gpu and GPU_AVAILABLE:
        return cp
    if use_gpu and not GPU_AVAILABLE:
        print("GPU requested, but CuPy/GPU is not available. Falling back to CPU.")
    return np

def sync_if_gpu(xp):
    if GPU_AVAILABLE and xp is cp:
        cp.cuda.Stream.null.synchronize()


def to_numpy(x):
    if GPU_AVAILABLE and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x


GAMMA = 2.0 - np.sqrt(2.0)


def make_alphas(d, use_gpu=False, dtype=np.float64):
    xp = get_xp(use_gpu)
    return xp.logspace(0.0, 6.0, d, dtype=dtype)


def exact_solution(alphas, t, use_gpu=False):
    xp = get_xp(use_gpu)
    return xp.exp(-alphas * t)


def euler_step(y, alphas, dt):
    return y - dt * alphas * y


def integrate_euler(alphas, dt, tf, use_gpu=False):
    xp = get_xp(use_gpu)
    nsteps = int(round(tf / dt))
    y = xp.ones_like(alphas)

    for _ in range(nsteps):
        y = euler_step(y, alphas, dt)

    return y


def tr_step(y, alphas, dt):
    numerator = 1.0 - 0.5 * dt * alphas
    denominator = 1.0 + 0.5 * dt * alphas
    return (numerator / denominator) * y


def integrate_tr(alphas, dt, tf, use_gpu=False):
    xp = get_xp(use_gpu)
    nsteps = int(round(tf / dt))
    y = xp.ones_like(alphas)

    for _ in range(nsteps):
        y = tr_step(y, alphas, dt)

    return y


def trbdf2_step(y, alphas, dt, gamma=GAMMA):

    num1 = 1.0 - 0.5 * gamma * dt * alphas
    den1 = 1.0 + 0.5 * gamma * dt * alphas
    Y = (num1 / den1) * y
    a = 1.0 / (gamma * (2.0 - gamma))
    b = -((1.0 - gamma) ** 2) / (gamma * (2.0 - gamma))
    c = (1.0 - gamma) / (2.0 - gamma)

    return (a * Y + b * y) / (1.0 + c * dt * alphas)


def integrate_trbdf2(alphas, dt, tf, use_gpu=False, gamma=GAMMA):
    xp = get_xp(use_gpu)
    nsteps = int(round(tf / dt))
    y = xp.ones_like(alphas)

    for _ in range(nsteps):
        y = trbdf2_step(y, alphas, dt, gamma=gamma)

    return y

def integrate_history(method_name, alphas_selected, dt, tf):
    nsteps = int(round(tf / dt))
    t_values = np.linspace(0.0, tf, nsteps + 1)
    y = np.ones_like(alphas_selected, dtype=np.float64)
    Yhist = np.zeros((alphas_selected.size, nsteps + 1), dtype=np.float64)
    Yhist[:, 0] = y

    for n in range(nsteps):
        if method_name == "TR":
            y = tr_step(y, alphas_selected, dt)
        elif method_name == "TRBDF2":
            y = trbdf2_step(y, alphas_selected, dt)
        elif method_name == "Euler":
            y = euler_step(y, alphas_selected, dt)
        else:
            raise ValueError("Unknown method_name")
        Yhist[:, n + 1] = y

    return t_values, Yhist


def max_relative_error(y_num, y_exact, tiny=1e-30):
    denom = np.maximum(np.abs(y_exact), tiny)
    rel = np.abs(y_num - y_exact) / denom
    return np.max(rel)


def plot_explicit_instability(filename=None):
    if filename is None:
        filename = os.path.join(SCRIPT_DIR, "problem4_explicit_instability.png")

    d_demo = 4000
    tf_demo = 0.01
    dt_stable = 1.0e-6
    dt_unstable = 5.0e-6

    alphas = np.logspace(0.0, 6.0, d_demo)
    y_exact = np.exp(-alphas * tf_demo)
    y_stable = integrate_euler(alphas, dt_stable, tf_demo, use_gpu=False)
    y_unstable = integrate_euler(alphas, dt_unstable, tf_demo, use_gpu=False)

    plt.figure(figsize=(8, 6))
    plt.semilogx(alphas, np.abs(y_exact), label=f"Exact at t={tf_demo}")
    plt.semilogx(alphas, np.abs(y_stable), label=f"Euler stable-ish dt={dt_stable:.0e}")
    plt.semilogx(alphas, np.abs(y_unstable), label=f"Euler unstable dt={dt_unstable:.0e}")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$|y(t_f)|$")
    plt.title("Problem 4: Explicit Euler Instability for Stiff Modes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

    print(f"Saved explicit-instability plot to: {filename}")

def plot_stiff_mode_damping(filename=None):
    if filename is None:
        filename = os.path.join(SCRIPT_DIR, "problem4_stiff_mode_damping.png")

    alphas_selected = np.array([1.0, 1.0e3, 1.0e6], dtype=np.float64)
    labels = [r"$\alpha=1$", r"$\alpha=10^3$", r"$\alpha=10^6$"]
    dt = 1.0e-2
    tf = 0.2

    t_tr, y_tr = integrate_history("TR", alphas_selected, dt, tf)
    t_tb, y_tb = integrate_history("TRBDF2", alphas_selected, dt, tf)

    plt.figure(figsize=(9, 6))

    for i, label in enumerate(labels):
        plt.semilogy(t_tr, np.abs(y_tr[i]) + 1e-300, "--", label=f"TR {label}")
        plt.semilogy(t_tb, np.abs(y_tb[i]) + 1e-300, "-", label=f"TRBDF2 {label}")

    plt.xlabel("t")
    plt.ylabel(r"$|y(t)|$")
    plt.title("Problem 4: TR vs TRBDF2 Damping of Stiff Modes")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

    print(f"Saved TR-vs-TRBDF2 damping plot to: {filename}")


def time_method(method_name, d, dt, tf, use_gpu=False, repeats=3, dtype=np.float32):
    xp = get_xp(use_gpu)
    best_time = float("inf")

    for _ in range(repeats):
        alphas = make_alphas(d, use_gpu=use_gpu, dtype=dtype)

        sync_if_gpu(xp)
        t1 = time.perf_counter()

        if method_name == "TR":
            y = integrate_tr(alphas, dt, tf, use_gpu=use_gpu)
        elif method_name == "TRBDF2":
            y = integrate_trbdf2(alphas, dt, tf, use_gpu=use_gpu)
        else:
            raise ValueError("Unknown method_name")

        sync_if_gpu(xp)
        t2 = time.perf_counter()

        elapsed = t2 - t1
        if elapsed < best_time:
            best_time = elapsed

    return best_time


def benchmark_trbdf2_gpu(d_values, dt=1e-2, tf=1.0, repeats=3, filename=None):
    if filename is None:
        filename = os.path.join(SCRIPT_DIR, "problem4_trbdf2_benchmark.txt")

    results = {
        "d": [],
        "cpu_time": [],
        "gpu_time": [],
        "speedup": []
    }

    print("\nTRBDF2 GPU Benchmark")
    print("-" * 65)

    if GPU_AVAILABLE:
        print(f"{'d':>12s} {'CPU Time (s)':>18s} {'GPU Time (s)':>18s} {'Speedup':>12s}")
    else:
        print(f"{'d':>12s} {'CPU Time (s)':>18s}")

    with open(filename, "w", encoding="utf-8") as file:
        file.write("d CPU_Time GPU_Time Speedup\n")

        for d in d_values:
            cpu_time = time_method("TRBDF2", d, dt, tf, use_gpu=False, repeats=repeats, dtype=np.float32)

            if GPU_AVAILABLE:
                gpu_time = time_method("TRBDF2", d, dt, tf, use_gpu=True, repeats=repeats, dtype=np.float32)
                speedup = cpu_time / gpu_time
                print(f"{d:12d} {cpu_time:18.6e} {gpu_time:18.6e} {speedup:12.4f}")
                file.write(f"{d} {cpu_time:.16e} {gpu_time:.16e} {speedup:.16e}\n")
            else:
                gpu_time = np.nan
                speedup = np.nan
                print(f"{d:12d} {cpu_time:18.6e}")
                file.write(f"{d} {cpu_time:.16e} nan nan\n")

            results["d"].append(d)
            results["cpu_time"].append(cpu_time)
            results["gpu_time"].append(gpu_time)
            results["speedup"].append(speedup)

    for key in results:
        results[key] = np.array(results[key])

    print(f"\nSaved TRBDF2 benchmark to: {filename}")
    return results


def plot_trbdf2_benchmark(results, filename=None):
    if filename is None:
        filename = os.path.join(SCRIPT_DIR, "problem4_trbdf2_runtime.png")

    logd = np.log10(results["d"])

    plt.figure(figsize=(8, 6))
    plt.plot(logd, results["cpu_time"], marker="o", label="TRBDF2 CPU")

    if GPU_AVAILABLE:
        plt.plot(logd, results["gpu_time"], marker="s", label="TRBDF2 GPU")

    plt.yscale("log")
    plt.xlabel(r"$\log_{10} d$")
    plt.ylabel("Runtime (s)")
    plt.title("Problem 4: TRBDF2 NumPy vs CuPy Runtime")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

    print(f"Saved TRBDF2 runtime plot to: {filename}")

def build_summary_table(d=200000, dt_values=(1e-3, 5e-2), tf=1.0, filename=None):
    if filename is None:
        filename = os.path.join(SCRIPT_DIR, "problem4_summary.txt")

    alphas = make_alphas(d, use_gpu=False, dtype=np.float64)
    y_exact = exact_solution(alphas, tf, use_gpu=False)

    rows = []

    print("\nSummary Table")
    print("-" * 115)
    print(f"{'Method':<10s} {'dt':>10s} {'Runtime(s)':>15s} {'Max Rel Err':>18s} {'Notes':<50s}")

    with open(filename, "w", encoding="utf-8") as file:
        file.write("Method dt Runtime_s MaxRelativeError Notes\n")

        for method_name in ["TR", "TRBDF2"]:
            for dt in dt_values:
                t1 = time.perf_counter()
                if method_name == "TR":
                    y_num = integrate_tr(alphas, dt, tf, use_gpu=False)
                else:
                    y_num = integrate_trbdf2(alphas, dt, tf, use_gpu=False)
                t2 = time.perf_counter()

                runtime = t2 - t1
                err = max_relative_error(y_num, y_exact)

                if method_name == "TR" and dt >= 1e-2:
                    notes = "A-stable, not L-stable; weak damping of very stiff modes"
                elif method_name == "TRBDF2" and dt >= 1e-2:
                    notes = "Better stiff damping; L-stable behavior on stiff modes"
                elif method_name == "TR":
                    notes = "Accurate for smaller dt"
                else:
                    notes = "Accurate and strongly damped"

                print(f"{method_name:<10s} {dt:10.2e} {runtime:15.6e} {err:18.6e} {notes:<50s}")
                file.write(f"{method_name} {dt:.6e} {runtime:.16e} {err:.16e} {notes}\n")

                rows.append((method_name, dt, runtime, err, notes))

    print(f"\nSaved summary table to: {filename}")
    return rows

if __name__ == "__main__":
    print("Problem 4: Stiff decay modes (TR vs TRBDF2)")
    print("=" * 65)
    print(f"Script directory: {SCRIPT_DIR}")

    if GPU_AVAILABLE:
        print("GPU is available through CuPy.")
    else:
        print("GPU is not available. Running CPU-only fallback.")

    plot_explicit_instability()

    plot_stiff_mode_damping()

    d_values = [10**3, 10**4, 10**5, 10**6]
    benchmark_results = benchmark_trbdf2_gpu(d_values=d_values, dt=1e-2, tf=1.0, repeats=3)
    plot_trbdf2_benchmark(benchmark_results)

    build_summary_table(d=200000, dt_values=(1e-3, 5e-2), tf=1.0)
