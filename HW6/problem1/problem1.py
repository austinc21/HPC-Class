import os
import time
import numpy as np
import matplotlib.pyplot as plt


# save everything in the folder where script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# detects GPU, if GPU is not present or CuPy is not installed, it will only use cpu
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


# Original ODE
def f(t, y):
    return -y

# Exact Solution with IC y(0) = 1 
def exact_solution(t):
    return np.exp(-t)


# Forward Euler eq 3.4 in numerical methods
def euler_step(f, t, y, dt):
    return y + dt * f(t, y)

# RK2 eq 3.24 in numerical methods
def rk2_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    return y + dt * k2

# RK4 eq 3.26 in numerical methods
def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# Generic solution
def solve_ivp(step_func, dt, use_gpu=False, N=1):
    xp = get_xp(use_gpu)

    t0 = 0.0
    tf = 1.0
    y0 = 1.0

    nsteps = int(round((tf - t0) / dt))

    t = t0
    y = xp.ones(N) * y0

    for _ in range(nsteps):
        y = step_func(f, t, y, dt)
        t += dt

    if xp is np:
        return y
    return cp.asnumpy(y)


#Validate cpu and gpu results
def validate_cpu_gpu(dt=2.0**(-8), N=1):
    methods = {
        "Euler": euler_step,
        "RK2": rk2_step,
        "RK4": rk4_step
    }

    print("\nCPU vs GPU validation")
    print("-" * 70)

    if not GPU_AVAILABLE: #skipping if no GPU
        print("GPU validation skipped: CuPy/GPU not available.")
        return

    print(f"{'Method':>8s} {'CPU Final Value':>20s} {'GPU Final Value':>20s} {'|diff|':>12s}")
    for name, method in methods.items():
        y_cpu = solve_ivp(method, dt, use_gpu=False, N=N)[0] #no gpu
        y_gpu = solve_ivp(method, dt, use_gpu=True, N=N)[0] # with gpu
        diff = abs(y_cpu - y_gpu)
        print(f"{name:>8s} {y_cpu:20.12e} {y_gpu:20.12e} {diff:12.4e}")


# convergence
def convergence_study(use_gpu=False, N=1, save_filename=None):
    if save_filename is None:
        save_filename = os.path.join(SCRIPT_DIR, "problem1_convergence.txt")

    methods = {
        "Euler": euler_step,
        "RK2": rk2_step,
        "RK4": rk4_step
    }

    ns = [4, 5, 6, 7, 8, 9, 10]
    dt_values = np.array([2.0**(-n) for n in ns])

    results = {}

    print("\nConvergence study")
    print("-" * 55)
    print(f"{'Method':>8s} {'dt':>12s} {'Error':>18s}")

    with open(save_filename, "w", encoding="utf-8") as file:
        file.write("Method dt Error\n")

        for method_name, method in methods.items():
            errors = []

            for dt in dt_values:
                y_num = solve_ivp(method, dt, use_gpu=use_gpu, N=N)[0]
                err = abs(y_num - exact_solution(1.0))
                errors.append(err)

                print(f"{method_name:>8s} {dt:12.5e} {err:18.10e}")
                file.write(f"{method_name} {dt:.16e} {err:.16e}\n")

            results[method_name] = {
                "dt": dt_values.copy(),
                "error": np.array(errors)
            }

    print(f"\nSaved convergence data to: {save_filename}")
    return results



def compute_orders(results, save_filename=None):
    if save_filename is None:
        save_filename = os.path.join(SCRIPT_DIR, "problem1_orders.txt")

    orders = {}

    print("\nObserved convergence orders")
    print("-" * 40)
    print(f"{'Method':>8s} {'Order':>12s}")

    with open(save_filename, "w", encoding="utf-8") as file:
        file.write("Method ObservedOrder Notes\n")

        for method_name, data in results.items():
            dt = data["dt"]
            err = data["error"]

            coeffs = np.polyfit(np.log(dt), np.log(err), 1)
            order = coeffs[0]
            orders[method_name] = order

            print(f"{method_name:>8s} {order:12.6f}")
            file.write(f"{method_name} {order:.8f} observed_from_loglog_fit\n")

    print(f"\nSaved observed orders to: {save_filename}")
    return orders


# timing
def time_method(step_func, dt, use_gpu=False, N=1, repeats=3):
    xp = get_xp(use_gpu)
    best_time = float("inf")

    for _ in range(repeats):
        sync_if_gpu(xp)
        t1 = time.perf_counter()
        solve_ivp(step_func, dt, use_gpu=use_gpu, N=N)
        sync_if_gpu(xp)
        t2 = time.perf_counter()

        elapsed = t2 - t1
        if elapsed < best_time:
            best_time = elapsed

    return best_time


def timing_study(N=1, repeats=3, save_filename=None):
    if save_filename is None:
        save_filename = os.path.join(SCRIPT_DIR, "problem1_timing.txt")

    methods = {
        "Euler": euler_step,
        "RK2": rk2_step,
        "RK4": rk4_step
    }

    ns = [4, 5, 6, 7, 8, 9, 10]
    dt_values = np.array([2.0**(-n) for n in ns])

    timing_results = {}

    print("\nTiming study")
    print("-" * 75)

    if GPU_AVAILABLE:
        print(f"{'Method':>8s} {'dt':>12s} {'CPU Time (s)':>18s} {'GPU Time (s)':>18s}")
    else:
        print(f"{'Method':>8s} {'dt':>12s} {'CPU Time (s)':>18s}")

    with open(save_filename, "w", encoding="utf-8") as file:
        file.write("Method dt CPU_Time GPU_Time Speedup\n")

        for method_name, method in methods.items():
            cpu_times = []
            gpu_times = []

            for dt in dt_values:
                cpu_time = time_method(method, dt, use_gpu=False, N=N, repeats=repeats)
                cpu_times.append(cpu_time)

                if GPU_AVAILABLE:
                    gpu_time = time_method(method, dt, use_gpu=True, N=N, repeats=repeats)
                    gpu_times.append(gpu_time)
                    speedup = cpu_time / gpu_time

                    print(f"{method_name:>8s} {dt:12.5e} {cpu_time:18.6e} {gpu_time:18.6e}")
                    file.write(f"{method_name} {dt:.16e} {cpu_time:.16e} {gpu_time:.16e} {speedup:.16e}\n")
                else:
                    print(f"{method_name:>8s} {dt:12.5e} {cpu_time:18.6e}")
                    file.write(f"{method_name} {dt:.16e} {cpu_time:.16e} nan nan\n")

            timing_results[method_name] = {
                "dt": dt_values.copy(),
                "cpu_time": np.array(cpu_times),
                "gpu_time": np.array(gpu_times) if GPU_AVAILABLE else None
            }

    print(f"\nSaved timing data to: {save_filename}")
    return timing_results


def print_smallest_dt_table(timing_results):
    print("\nSmallest-dt timing summary")
    print("-" * 70)

    if GPU_AVAILABLE:
        print(f"{'Method':>8s} {'CPU Time (s)':>18s} {'GPU Time (s)':>18s} {'Speedup':>12s}")
        for method_name, data in timing_results.items():
            cpu_time = data["cpu_time"][-1]
            gpu_time = data["gpu_time"][-1]
            speedup = cpu_time / gpu_time
            print(f"{method_name:>8s} {cpu_time:18.6e} {gpu_time:18.6e} {speedup:12.4f}")
    else:
        print(f"{'Method':>8s} {'CPU Time (s)':>18s}")
        for method_name, data in timing_results.items():
            cpu_time = data["cpu_time"][-1]
            print(f"{method_name:>8s} {cpu_time:18.6e}")


#convergence plot
def plot_convergence(results, filename=None):
    if filename is None:
        filename = os.path.join(SCRIPT_DIR, "problem1_error_plot.png")

    plt.figure(figsize=(8, 6))

    for method_name, data in results.items():
        dt = data["dt"]
        err = data["error"]
        plt.loglog(dt, err, marker="o", label=method_name)

    dt_ref = results["Euler"]["dt"]

    c1 = results["Euler"]["error"][0] / (dt_ref[0]**1)
    c2 = results["RK2"]["error"][0] / (dt_ref[0]**2)
    c4 = results["RK4"]["error"][0] / (dt_ref[0]**4)

    plt.loglog(dt_ref, c1 * dt_ref**1, "--", label="Order 1 ref")
    plt.loglog(dt_ref, c2 * dt_ref**2, "--", label="Order 2 ref")
    plt.loglog(dt_ref, c4 * dt_ref**4, "--", label="Order 4 ref")

    plt.xlabel(r"$\Delta t$")
    plt.ylabel(r"Global error at $t=1$")
    plt.title("Problem 1: Error vs. Time Step")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

    print(f"Saved convergence plot to: {filename}")


#plot timing
def plot_timing(timing_results, filename=None):
    if filename is None:
        filename = os.path.join(SCRIPT_DIR, "problem1_timing_plot.png")

    plt.figure(figsize=(8, 6))

    for method_name, data in timing_results.items():
        dt = data["dt"]
        plt.loglog(dt, data["cpu_time"], marker="o", label=f"{method_name} CPU")

        if GPU_AVAILABLE and data["gpu_time"] is not None:
            plt.loglog(dt, data["gpu_time"], marker="s", label=f"{method_name} GPU")

    plt.xlabel(r"$\Delta t$")
    plt.ylabel("Computation time (s)")
    plt.title("Problem 1: Computation Time vs. Time Step")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

    print(f"Saved timing plot to: {filename}")


if __name__ == "__main__":
    print("Problem 1: Explicit solver verification")
    print("=" * 50)
    print(f"Script directory: {SCRIPT_DIR}")

    if GPU_AVAILABLE:
        print("GPU is available through CuPy.")
    else:
        print("GPU is not available. Running CPU-only version.")

    N = 1

    validate_cpu_gpu(dt=2.0**(-8), N=N)

    convergence_results = convergence_study(use_gpu=False, N=N)
    observed_orders = compute_orders(convergence_results)
    timing_results = timing_study(N=N, repeats=3)

    print_smallest_dt_table(timing_results)

    plot_convergence(convergence_results)
    plot_timing(timing_results)