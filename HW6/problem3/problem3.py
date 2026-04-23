import os
import time
import numpy as np
import matplotlib.pyplot as plt



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


#gpu detection
try:
    import cupy as cp
    try:
        _ = cp.arange(1)
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


def f(y, r):
    return r * y * (1.0 - y)


def rk4_step_batch(y, dt, r):
    k1 = f(y, r)
    k2 = f(y + 0.5 * dt * k1, r)
    k3 = f(y + 0.5 * dt * k2, r)
    k4 = f(y + dt * k3, r)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_batch_rk4(N, dt=1e-3, tf=10.0, r=2.0, use_gpu=False):
    xp = get_xp(use_gpu)

    nsteps = int(round(tf / dt))
    y0 = xp.linspace(0.01, 0.99, N)
    y = y0.copy()

    for _ in range(nsteps):
        y = rk4_step_batch(y, dt, r)

    if xp is np:
        return y
    return cp.asnumpy(y)

def time_integration(N, dt=1e-3, tf=10.0, r=2.0, use_gpu=False, repeats=3):
    xp = get_xp(use_gpu)
    best_time = float("inf")

    for _ in range(repeats):
        sync_if_gpu(xp)
        t1 = time.perf_counter()
        integrate_batch_rk4(N, dt=dt, tf=tf, r=r, use_gpu=use_gpu)
        sync_if_gpu(xp)
        t2 = time.perf_counter()
        elapsed = t2 - t1

        if elapsed < best_time:
            best_time = elapsed

    return best_time

def validate_cpu_gpu(N=1000, dt=1e-3, tf=10.0, r=2.0):
    print("\nCPU vs GPU validation")
    print("-" * 60)

    y_cpu = integrate_batch_rk4(N, dt=dt, tf=tf, r=r, use_gpu=False)

    if GPU_AVAILABLE:
        y_gpu = integrate_batch_rk4(N, dt=dt, tf=tf, r=r, use_gpu=True)
        max_diff = np.max(np.abs(y_cpu - y_gpu))
        print(f"N = {N}")
        print(f"max |CPU - GPU| = {max_diff:.6e}")
    else:
        print("GPU validation skipped: GPU not available.")


def benchmark_problem3(
    N_values,
    dt=1e-3,
    tf=10.0,
    r=2.0,
    repeats=3,
    save_filename=None
):
    if save_filename is None:
        save_filename = os.path.join(SCRIPT_DIR, "problem3_benchmark.txt")

    results = {
        "N": [],
        "cpu_time": [],
        "gpu_time": [],
        "speedup": [],
        "cpu_throughput": [],
        "gpu_throughput": []
    }

    nsteps = int(round(tf / dt))

    print("\nBenchmark results")
    print("-" * 95)

    if GPU_AVAILABLE:
        print(f"{'N':>12s} {'CPU Time (s)':>18s} {'GPU Time (s)':>18s} {'Speedup':>12s} {'CPU Thru':>16s} {'GPU Thru':>16s}")
    else:
        print(f"{'N':>12s} {'CPU Time (s)':>18s} {'CPU Thru':>16s}")

    with open(save_filename, "w", encoding="utf-8") as file:
        file.write("N CPU_Time GPU_Time Speedup CPU_Throughput GPU_Throughput\n")

        for N in N_values:
            cpu_time = time_integration(N, dt=dt, tf=tf, r=r, use_gpu=False, repeats=repeats)
            cpu_throughput = (N * nsteps) / cpu_time

            if GPU_AVAILABLE:
                gpu_time = time_integration(N, dt=dt, tf=tf, r=r, use_gpu=True, repeats=repeats)
                gpu_throughput = (N * nsteps) / gpu_time
                speedup = cpu_time / gpu_time

                print(f"{N:12d} {cpu_time:18.6e} {gpu_time:18.6e} {speedup:12.4f} {cpu_throughput:16.6e} {gpu_throughput:16.6e}")
                file.write(f"{N} {cpu_time:.16e} {gpu_time:.16e} {speedup:.16e} {cpu_throughput:.16e} {gpu_throughput:.16e}\n")
            else:
                gpu_time = np.nan
                gpu_throughput = np.nan
                speedup = np.nan

                print(f"{N:12d} {cpu_time:18.6e} {cpu_throughput:16.6e}")
                file.write(f"{N} {cpu_time:.16e} nan nan {cpu_throughput:.16e} nan\n")

            results["N"].append(N)
            results["cpu_time"].append(cpu_time)
            results["gpu_time"].append(gpu_time)
            results["speedup"].append(speedup)
            results["cpu_throughput"].append(cpu_throughput)
            results["gpu_throughput"].append(gpu_throughput)

    for key in results:
        results[key] = np.array(results[key])

    print(f"\nSaved benchmark data to: {save_filename}")
    return results


# plotting runtimes
def plot_runtime(results, filename=None):
    if filename is None:
        filename = os.path.join(SCRIPT_DIR, "problem3_runtime.png")

    logN = np.log10(results["N"])

    plt.figure(figsize=(8, 6))
    plt.plot(logN, results["cpu_time"], marker="o", label="CPU")

    if GPU_AVAILABLE:
        plt.plot(logN, results["gpu_time"], marker="s", label="GPU")

    plt.yscale("log")
    plt.xlabel(r"$\log_{10} N$")
    plt.ylabel("Runtime (s)")
    plt.title("Problem 3: CPU vs GPU Runtime")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

    print(f"Saved runtime plot to: {filename}")

# plotting speedup
def plot_speedup(results, filename=None):
    if filename is None:
        filename = os.path.join(SCRIPT_DIR, "problem3_speedup.png")

    if not GPU_AVAILABLE:
        print("Skipping speedup plot: GPU not available.")
        return

    logN = np.log10(results["N"])

    plt.figure(figsize=(8, 6))
    plt.plot(logN, results["speedup"], marker="o")
    plt.xlabel(r"$\log_{10} N$")
    plt.ylabel(r"Speedup $S(N)=t_{CPU}/t_{GPU}$")
    plt.title("Problem 3: GPU Speedup")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

    print(f"Saved speedup plot to: {filename}")


# plotting throughput
def plot_throughput(results, filename=None):
    if filename is None:
        filename = os.path.join(SCRIPT_DIR, "problem3_throughput.png")

    logN = np.log10(results["N"])

    plt.figure(figsize=(8, 6))
    plt.plot(logN, results["cpu_throughput"], marker="o", label="CPU")

    if GPU_AVAILABLE:
        plt.plot(logN, results["gpu_throughput"], marker="s", label="GPU")

    plt.yscale("log")
    plt.xlabel(r"$\log_{10} N$")
    plt.ylabel("Throughput (trajectories * steps / s)")
    plt.title("Problem 3: Throughput")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

    print(f"Saved throughput plot to: {filename}")



if __name__ == "__main__":
    print("Problem 3: GPU speedup via large-batch ensemble integration")
    print("=" * 70)
    print(f"Script directory: {SCRIPT_DIR}")

    if GPU_AVAILABLE:
        print("GPU is available through CuPy.")
    else:
        print("GPU is not available. Running CPU-only fallback.")

    dt = 1e-3
    tf = 10.0
    r = 2.0

    N_values = [10**3, 10**4, 10**5, 10**6]

    validate_cpu_gpu(N=1000, dt=dt, tf=tf, r=r)

    results = benchmark_problem3(
        N_values=N_values,
        dt=dt,
        tf=tf,
        r=r,
        repeats=3
    )

    plot_runtime(results)
    plot_speedup(results)
    plot_throughput(results)