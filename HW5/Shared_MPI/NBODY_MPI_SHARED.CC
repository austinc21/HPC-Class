#include <iostream>   // Standard I/O
#include <fstream>    // File I/O
#include <random>     // Random number generators
#include <vector>     // Vector (dynamic array)
#include <chrono>     // Time utilities
#include <string>     // std::to_string
#include <cmath>      // std::sqrt
#include <cstdlib>    // std::atoi
#include <mpi.h>      // MPI (parallelism)

// Global constants
static int N = 128;                                // Number of masses
static const int D = 3;                            // Dimensionality
static int ND = N * D;                             // Size of the state vectors
static const double G = 0.5;                       // Gravitational constant
static const double dt = 1e-3;                     // Time step size
static const int T = 300;                          // Number of time steps
static const double t_max = static_cast<double>(T) * dt; // Maximum time
static const double x_min = 0.;                    // Minimum position
static const double x_max = 1.;                    // Maximum position
static const double v_min = 0.;                    // Minimum velocity
static const double v_max = 0.;                    // Maximum velocity
static const double m_0 = 1.;                      // Mass value
static const double epsilon = 0.01;                // Softening parameter
static const double epsilon2 = epsilon * epsilon;  // Softening parameter^2

using Vec = std::vector<double>;

// Random number generator
static std::mt19937 gen;
static std::uniform_real_distribution<> ran(0., 1.);

// MPI globals
static int rank, n_ranks;                 // Process rank and number of processes
static std::vector<int> counts, displs;   // Counts and displacements
static std::vector<int> countsD, displsD; // State counts and displacements
static int N_beg, N_end, N_local;         // Mass range for each process [N_beg, N_end)
static int ND_beg, ND_end, ND_local;      // State vector range for each process [ND_beg, ND_end)

// Shared memory for masses, positions, velocities, and accelerations
static double *m, *x, *v, *a, *x_next, *v_next;
static MPI_Win win_m, win_x, win_v, win_a, win_x_next, win_v_next;

// Print a vector to a file
template <typename T>
void save(const std::vector<T>& vec, const std::string& filename,
          const std::string& header = "") {
    std::ofstream file(filename);
    if (file.is_open()) {
        if (!header.empty())
            file << "# " << header << std::endl;
        for (const auto& elem : vec)
            file << elem << " ";
        file << std::endl;
        file.close();
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

// Set up parallelism
void setup_parallelism() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    auto now = std::chrono::high_resolution_clock::now();
    auto now_cast = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    auto now_int = now_cast.time_since_epoch().count();

    gen.seed(static_cast<unsigned int>(now_int ^ rank));

    counts.resize(n_ranks);
    displs.resize(n_ranks);
    countsD.resize(n_ranks);
    displsD.resize(n_ranks);

    const int remainder = N % n_ranks;
    for (int i = 0; i < n_ranks; ++i) {
        counts[i] = N / n_ranks;
        displs[i] = i * counts[i];

        if (i < remainder) {
            counts[i] += 1;
            displs[i] += i;
        } else {
            displs[i] += remainder;
        }

        countsD[i] = counts[i] * D;
        displsD[i] = displs[i] * D;
    }

    N_beg = displs[rank];
    N_end = N_beg + counts[rank];
    ND_beg = N_beg * D;
    ND_end = N_end * D;
    N_local = N_end - N_beg;
    ND_local = ND_end - ND_beg;

    // Allocate shared memory for positions, velocities, and accelerations
    if (rank == 0) {
        MPI_Win_allocate_shared(N * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &m, &win_m);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &x, &win_x);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &v, &win_v);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &a, &win_a);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &x_next, &win_x_next);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &v_next, &win_v_next);
    } else {
        int disp_unit;
        MPI_Aint size;

        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &m, &win_m);
        MPI_Win_shared_query(win_m, 0, &size, &disp_unit, &m);

        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &x, &win_x);
        MPI_Win_shared_query(win_x, 0, &size, &disp_unit, &x);

        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &v, &win_v);
        MPI_Win_shared_query(win_v, 0, &size, &disp_unit, &v);

        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &a, &win_a);
        MPI_Win_shared_query(win_a, 0, &size, &disp_unit, &a);

        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &x_next, &win_x_next);
        MPI_Win_shared_query(win_x_next, 0, &size, &disp_unit, &x_next);

        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &v_next, &win_v_next);
        MPI_Win_shared_query(win_v_next, 0, &size, &disp_unit, &v_next);
    }
}

// Generate random initial conditions for N masses
void initial_conditions() {
    const double dx = x_max - x_min;
    const double dv = v_max - v_min;

    for (int i = 0; i < N; ++i) {
        m[i] = m_0;
    }

    for (int i = ND_beg; i < ND_end; ++i) {
        x[i] = ran(gen) * dx + x_min;
        v[i] = ran(gen) * dv + v_min;
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
}

// Compute the acceleration of local masses
void acceleration() {
    for (int i = ND_beg; i < ND_end; ++i) {
        a[i] = 0.0;
    }

    for (int i = N_beg; i < N_end; ++i) {
        const int iD = i * D;
        double dx[D];

        for (int j = 0; j < N; ++j) {
            const int jD = j * D;
            double dx2 = epsilon2;

            for (int k = 0; k < D; ++k) {
                dx[k] = x[jD + k] - x[iD + k];
                dx2 += dx[k] * dx[k];
            }

            const double Gm_dx3 = G * m[j] / (dx2 * std::sqrt(dx2));

            for (int k = 0; k < D; ++k) {
                a[iD + k] += Gm_dx3 * dx[k];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
}

// Compute the next position and velocity for local masses
void timestep() {
    acceleration();

    for (int i = ND_beg; i < ND_end; ++i) {
        v_next[i] = a[i] * dt + v[i];
        x_next[i] = v_next[i] * dt + x[i];
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes

    std::swap(x, x_next); // Update the positions and velocities
    std::swap(v, v_next); // By swapping the pointers to the arrays

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
}

// Calculate the local kinetic energy
double kinetic_energy() {
    double KE_local = 0.0;

    for (int i = N_beg; i < N_end; ++i) {
        double v2 = 0.0;
        for (int j = 0; j < D; ++j) {
            const int k = i * D + j;
            v2 += v[k] * v[k];
        }
        KE_local += 0.5 * m[i] * v2;
    }

    return KE_local;
}

// Main function
int main(int argc, char** argv) {
    if (argc > 1) {
        N = std::atoi(argv[1]);
        ND = N * D;
    }

    setup_parallelism();

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    Vec t(T + 1);
    for (int i = 0; i <= T; ++i) {
        t[i] = double(i) * dt;
    }

    initial_conditions();

    Vec KE(T + 1, 0.0);
    KE[0] = kinetic_energy(); // Kinetic energy

    for (int n = 0; n < T; ++n) {
        timestep();
        KE[n + 1] = kinetic_energy(); // Kinetic energy
    }

    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, KE.data(), T + 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(KE.data(), nullptr, T + 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.;

    if (rank == 0) {
        save(KE, "KE_" + std::to_string(N) + ".txt", "Kinetic Energy");
        save(t, "time_" + std::to_string(N) + ".txt", "Time");

        std::cout << "Total Kinetic Energy = [" << KE[0];
        const int T_skip = T / 50;
        for (int n = 1; n <= T; n += T_skip) {
            std::cout << ", " << KE[n];
        }
        std::cout << "]" << std::endl;

        std::cout << "Runtime = " << elapsed << " s for N = " << N
                  << " using " << n_ranks << " MPI ranks (shared memory)" << std::endl;
    }

    // Positions, velocities, and accelerations (reversed order)
    MPI_Win_free(&win_v_next);
    MPI_Win_free(&win_x_next);
    MPI_Win_free(&win_a);
    MPI_Win_free(&win_v);
    MPI_Win_free(&win_x);
    MPI_Win_free(&win_m);

    MPI_Finalize();
    return 0;
}