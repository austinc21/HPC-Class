import pandas as pd
import matplotlib.pyplot as plt

# load files
df = pd.read_csv("results/runtime.csv")
mpi = pd.read_csv("HW4/test_cases/mpi_runtime.csv")

# keep bins = 100
df = df[df["bins"] == 100]
mpi = mpi[mpi["bins"] == 100]

methods = sorted(df["method"].unique())

# -------- Runtime Plot --------
plt.figure(figsize=(8,5))

for method in methods:
    sub = df[df["method"] == method].sort_values("workers")
    plt.plot(sub["workers"], sub["runtime"], marker="o", label=method)

# MPI overlay
mpi = mpi.sort_values("workers")
plt.plot(mpi["workers"], mpi["runtime"], marker="o", linewidth=3, label="mpi")

plt.xlabel("Workers")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs Workers (bins=100)")
plt.grid(True)
plt.legend()

plt.savefig("runtime_methods_bins100.png")
plt.show()


# -------- Speedup Plot --------
plt.figure(figsize=(8,5))

for method in methods:
    sub = df[df["method"] == method].sort_values("workers")
    T1 = sub[sub["workers"] == 1]["runtime"].values[0]
    speedup = T1 / sub["runtime"]
    plt.plot(sub["workers"], speedup, marker="o", label=method)

# MPI speedup
T1 = mpi[mpi["workers"] == 1]["runtime"].values[0]
speedup = T1 / mpi["runtime"]
plt.plot(mpi["workers"], speedup, marker="o", linewidth=3, label="mpi")


plt.xlabel("Workers")
plt.ylabel("Speedup")
plt.title("Speedup vs Workers (bins=100)")
plt.grid(True)
plt.legend()

plt.savefig("speedup_methods_bins100.png")
plt.show()


# -------- Efficiency Plot --------
plt.figure(figsize=(8,5))

for method in methods:
    sub = df[df["method"] == method].sort_values("workers")
    T1 = sub[sub["workers"] == 1]["runtime"].values[0]
    speedup = T1 / sub["runtime"]
    efficiency = speedup / sub["workers"]
    plt.plot(sub["workers"], efficiency, marker="o", label=method)

# MPI efficiency
T1 = mpi[mpi["workers"] == 1]["runtime"].values[0]
speedup = T1 / mpi["runtime"]
efficiency = speedup / mpi["workers"]
plt.plot(mpi["workers"], efficiency, marker="o", linewidth=3, label="mpi")

plt.xlabel("Workers")
plt.ylabel("Efficiency")
plt.title("Efficiency vs Workers (bins=100)")
plt.grid(True)
plt.legend()

plt.savefig("efficiency_methods_bins100.png")
plt.show()