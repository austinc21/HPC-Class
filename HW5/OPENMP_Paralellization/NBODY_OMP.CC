#include <iostream>   // Standard I/O
#include <fstream>    // File I/O
#include <random>     // Random number generators
#include <vector>     // Vector (dynamic array)
#include <tuple>      // Tuple (multiple return values)
#include <chrono>     // Time utilities
#include <string>     // std::to_string
#include <cmath>      // sqrt
#include <cstdlib>    // std::atoi
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

// Generate random initial conditions for N masses
std::tuple<Vec, Vec> initial_conditions() {
    Vec x(ND), v(ND);
    const double dx = x_max - x_min;
    const double dv = v_max - v_min;

    #pragma omp parallel
    {
        std::mt19937 gen(omp_get_thread_num() + 1);
        std::uniform_real_distribution<> ran(0., 1.);

        #pragma omp for
        for (int i = 0; i < ND; ++i) {
            x[i] = ran(gen) * dx + x_min;
            v[i] = ran(gen) * dv + v_min;
        }
    }

    return {x, v};
}

// Compute the acceleration of all masses
Vec acceleration(const Vec& x, const Vec& m) {
    Vec a(ND, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
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
                const int iDk = iD + k;
                a[iDk] += Gm_dx3 * dx[k];
            }
        }
    }

    return a;
}

// Compute the next position and velocity for all masses
std::tuple<Vec, Vec> timestep(const Vec& x0, const Vec& v0, const Vec& m) {
    Vec a0 = acceleration(x0, m);
    Vec x1(ND), v1(ND);

    #pragma omp parallel for
    for (int i = 0; i < ND; ++i) {
        v1[i] = a0[i] * dt + v0[i];
        x1[i] = v1[i] * dt + x0[i];
    }

    return {x1, v1};
}

// Main function
int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();

    if (argc > 1) {
        N = std::atoi(argv[1]);
        ND = N * D;
    }

    Vec t(T + 1);
    for (int i = 0; i <= T; ++i)
        t[i] = double(i) * dt;

    Vec m(N, m_0);
    Vecs x(T + 1), v(T + 1);
    std::tie(x[0], v[0]) = initial_conditions();

    for (int n = 0; n < T; ++n)
        std::tie(x[n + 1], v[n + 1]) = timestep(x[n], v[n], m);

    Vec KE(T + 1);
    for (int n = 0; n <= T; ++n) {
        double KE_n = 0.;
        auto& v_n = v[n];

        #pragma omp parallel for reduction(+:KE_n)
        for (int i = 0; i < N; ++i) {
            double v2 = 0.;
            for (int j = 0; j < D; ++j) {
                const int k = i * D + j;
                v2 += v_n[k] * v_n[k];
            }
            KE_n += 0.5 * m[i] * v2;
        }

        KE[n] = KE_n;
    }

    save(KE, "KE_" + std::to_string(N) + ".txt", "Kinetic Energy");
    save(t, "time_" + std::to_string(N) + ".txt", "Time");

    std::cout << "Total Kinetic Energy = [" << KE[0];
    const int T_skip = T / 50;
    for (int n = 1; n <= T; n += T_skip)
        std::cout << ", " << KE[n];
    std::cout << "]" << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.;

    std::cout << "Runtime = " << elapsed << " s for N = " << N << std::endl;

    return 0;
}