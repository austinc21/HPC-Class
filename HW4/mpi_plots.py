import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("HW4/test_cases/mpi_runtime.csv")

bins_list = sorted(df["bins"].unique())

# -------- Runtime Plot --------
plt.figure()

for b in bins_list:
    subset = df[df["bins"] == b].sort_values("workers")
    plt.plot(subset["workers"], subset["runtime"], marker='o', label=f"bins={b}")

plt.xlabel("Workers")
plt.ylabel("Runtime (seconds)")
plt.title("MPI Runtime Scaling")
plt.legend()
plt.grid(True)

plt.savefig("runtime_vs_workers.png")
plt.show()


# -------- Speedup Plot --------
plt.figure()

for b in bins_list:
    subset = df[df["bins"] == b].sort_values("workers")
    
    T1 = subset[subset["workers"] == 1]["runtime"].values[0]
    speedup = T1 / subset["runtime"]
    
    plt.plot(subset["workers"], speedup, marker='o', label=f"bins={b}")

plt.xlabel("Workers")
plt.ylabel("Speedup")
plt.title("MPI Speedup")
plt.legend()
plt.grid(True)

plt.savefig("speedup.png")
plt.show()


# -------- Efficiency Plot --------
plt.figure()

for b in bins_list:
    subset = df[df["bins"] == b].sort_values("workers")
    
    T1 = subset[subset["workers"] == 1]["runtime"].values[0]
    speedup = T1 / subset["runtime"]
    efficiency = speedup / subset["workers"]
    
    plt.plot(subset["workers"], efficiency, marker='o', label=f"bins={b}")

plt.xlabel("Workers")
plt.ylabel("Efficiency")
plt.title("MPI Efficiency")
plt.legend()
plt.grid(True)

plt.savefig("efficiency.png")
plt.show()