#include <iostream>   // Standard I/O
#include <fstream>    // File I/O
#include <random>     // Random number generators
#include <vector>     // Vector (dynamic array)
#include <tuple>      // Tuple (multiple return values)
#include <chrono>     // Time utilities
#include <string>     // std::to_string
#include <cmath>      // std::sqrt
#include <cstdlib>    // std::atoi
#include <mpi.h>      // MPI (parallelism)
#include <omp.h>      // OpenMP (parallelism)

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
using Vecs = std::vector<Vec>;

// MPI globals
static int rank, n_ranks;                 // Process rank and number of processes
static std::vector<int> counts, displs;   // Counts and displacements
static std::vector<int> countsD, displsD; // State counts and displacements
static int N_beg, N_end, N_local;         // Mass range for each process [N_beg, N_end)
static int ND_beg, ND_end, ND_local;      // State vector range for each process [ND_beg, ND_end)

// RNG seed base for hybrid MPI+OpenMP
static unsigned long long now_int = 0;

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

// Set up MPI
void setup_parallelism() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    auto now = std::chrono::high_resolution_clock::now();
    auto now_cast = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    now_int = static_cast<unsigned long long>(now_cast.time_since_epoch().count());

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
}

// Generate random initial conditions for N masses
std::tuple<Vec, Vec> initial_conditions() {
    Vec x(ND, 0.0);
    Vec v(ND, 0.0);

    const double dx = x_max - x_min;
    const double dv = v_max - v_min;

    Vec x_local(ND_local);
    Vec v_local(ND_local);

    #pragma omp parallel
    {
        int thread = omp_get_thread_num();
        std::mt19937 gen(static_cast<unsigned int>(now_int ^ (thread * n_ranks + rank)));
        std::uniform_real_distribution<> ran(0., 1.);

        #pragma omp for
        for (int i = 0; i < ND_local; ++i) {
            x_local[i] = ran(gen) * dx + x_min;
            v_local[i] = ran(gen) * dv + v_min;
        }
    }

    MPI_Allgatherv(x_local.data(), ND_local, MPI_DOUBLE,
                   x.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Allgatherv(v_local.data(), ND_local, MPI_DOUBLE,
                   v.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    return {x, v};
}

// Compute the acceleration of local masses
Vec acceleration(const Vec& x, const Vec& m) {
    Vec a_local(ND_local, 0.0);

    #pragma omp parallel for
    for (int i = N_beg; i < N_end; ++i) {
        const int i_localD = (i - N_beg) * D;
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
                a_local[i_localD + k] += Gm_dx3 * dx[k];
            }
        }
    }

    return a_local;
}

// Compute the next position and velocity for all masses
std::tuple<Vec, Vec> timestep(const Vec& x0, const Vec& v0, const Vec& m) {
    Vec a_local = acceleration(x0, m);

    Vec x_local(ND_local), v_local(ND_local);
    Vec x1(ND, 0.0), v1(ND, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < ND_local; ++i) {
        const int global_i = ND_beg + i;
        v_local[i] = a_local[i] * dt + v0[global_i];
        x_local[i] = v_local[i] * dt + x0[global_i];
    }

    MPI_Allgatherv(x_local.data(), ND_local, MPI_DOUBLE,
                   x1.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Allgatherv(v_local.data(), ND_local, MPI_DOUBLE,
                   v1.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    return {x1, v1};
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
    for (int i = 0; i <= T; ++i)
        t[i] = double(i) * dt;

    Vec m(N, m_0);
    Vecs x(T + 1), v(T + 1);
    std::tie(x[0], v[0]) = initial_conditions();

    for (int n = 0; n < T; ++n)
        std::tie(x[n + 1], v[n + 1]) = timestep(x[n], v[n], m);

    Vec KE(T + 1, 0.0);
    for (int n = 0; n <= T; ++n) {
        double KE_n = 0.;
        auto& v_n = v[n];

        #pragma omp parallel for reduction(+:KE_n)
        for (int i = N_beg; i < N_end; ++i) {
            double v2 = 0.;
            for (int j = 0; j < D; ++j) {
                const int k = i * D + j;
                v2 += v_n[k] * v_n[k];
            }
            KE_n += 0.5 * m[i] * v2;
        }

        KE[n] = KE_n;
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
        for (int n = 1; n <= T; n += T_skip)
            std::cout << ", " << KE[n];
        std::cout << "]" << std::endl;

        std::cout << "Runtime = " << elapsed << " s for N = " << N
                  << " using " << n_ranks << " MPI ranks and "
                  << omp_get_max_threads() << " OpenMP threads per rank" << std::endl;
    }

    MPI_Finalize();
    return 0;
}