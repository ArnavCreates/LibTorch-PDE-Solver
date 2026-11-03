#include "pde_solver/fdm_gauss_seidel.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace pde_solver {

OpenMPRedBlackGaussSeidelSolver::OpenMPRedBlackGaussSeidelSolver(PDEDomain domain, FDMConfig config)
    : domain_(domain), config_(config) {}

FDMResult OpenMPRedBlackGaussSeidelSolver::solve(bool verbose) {
    FDMResult result;
    result.total_time_steps = config_.nt;
    result.num_threads_used = omp_get_max_threads();

    const int nx = config_.nx;
    const int ny = config_.ny;
    const int total_pts = nx * ny;

    const double dx = (domain_.x_max - domain_.x_min) / (nx - 1);
    const double dy = (domain_.y_max - domain_.y_min) / (ny - 1);
    const double dt = (domain_.t_max - domain_.t_min) / config_.nt;

    result.x_coords.resize(nx);
    for (int i = 0; i < nx; ++i) {
        result.x_coords[i] = domain_.x_min + i * dx;
    }

    result.y_coords.resize(ny);
    for (int j = 0; j < ny; ++j) {
        result.y_coords[j] = domain_.y_min + j * dy;
    }

    // Allocate solution grids (u_old is u^n, u is u^{n+1})
    std::vector<double> u_old(total_pts, 0.0);
    std::vector<double> u(total_pts, 0.0);

    // Initial Condition at t = 0: u(x, y, 0) = sin(pi * x) * sin(pi * y)
    for (int i = 0; i < nx; ++i) {
        double x = result.x_coords[i];
        for (int j = 0; j < ny; ++j) {
            double y = result.y_coords[j];
            u_old[i * ny + j] = std::sin(PI * x) * std::sin(PI * y);
            u[i * ny + j] = u_old[i * ny + j];
        }
    }

    const double rx = domain_.alpha * dt / (dx * dx);
    const double ry = domain_.alpha * dt / (dy * dy);
    const double denom = 1.0 + 2.0 * rx + 2.0 * ry;

    if (verbose) {
        std::cout << "========================================================\n"
                  << "  Starting OpenMP Red-Black Gauss-Seidel FDM Solver\n"
                  << "  Grid: " << nx << " x " << ny << " (" << total_pts << " pts)"
                  << ", Time Steps: " << config_.nt << ", Threads: " << result.num_threads_used << "\n"
                  << "  dx: " << dx << ", dy: " << dy << ", dt: " << dt
                  << ", rx: " << rx << ", ry: " << ry << "\n"
                  << "========================================================\n";
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    int64_t accumulated_iterations = 0;
    double max_final_residual = 0.0;

    for (int n = 0; n < config_.nt; ++n) {
        int iter = 0;
        double max_update = 1.0;

        while (iter < config_.max_iterations && max_update > config_.tolerance) {
            double max_diff_red = 0.0;

            // 1. Red update: (i + j) % 2 == 0
            #pragma omp parallel for collapse(2) reduction(max:max_diff_red) schedule(static)
            for (int i = 1; i < nx - 1; ++i) {
                for (int j = 1; j < ny - 1; ++j) {
                    if ((i + j) % 2 == 0) {
                        int idx = i * ny + j;
                        double u_prev = u[idx];
                        double u_new = (u_old[idx] +
                                        rx * (u[(i - 1) * ny + j] + u[(i + 1) * ny + j]) +
                                        ry * (u[i * ny + (j - 1)] + u[i * ny + (j + 1)])) / denom;
                        u[idx] = u_new;
                        max_diff_red = std::max(max_diff_red, std::abs(u_new - u_prev));
                    }
                }
            }

            // 2. Black update: (i + j) % 2 == 1, compute maximum difference
            double max_diff_black = 0.0;
            #pragma omp parallel for collapse(2) reduction(max:max_diff_black) schedule(static)
            for (int i = 1; i < nx - 1; ++i) {
                for (int j = 1; j < ny - 1; ++j) {
                    if ((i + j) % 2 == 1) {
                        int idx = i * ny + j;
                        double u_prev = u[idx];
                        double u_new = (u_old[idx] +
                                        rx * (u[(i - 1) * ny + j] + u[(i + 1) * ny + j]) +
                                        ry * (u[i * ny + (j - 1)] + u[i * ny + (j + 1)])) / denom;
                        u[idx] = u_new;
                        double diff = std::abs(u_new - u_prev);
                        max_diff_black = std::max(max_diff_black, diff);
                    }
                }
            }

            const double max_diff = std::max(max_diff_red, max_diff_black);
            max_update = max_diff;
            iter++;
        }

        accumulated_iterations += iter;
        max_final_residual = std::max(max_final_residual, max_update);

        // Update u_old for next time step
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < total_pts; ++k) {
            u_old[k] = u[k];
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.wallclock_time_sec = std::chrono::duration<double>(end_time - start_time).count();
    result.total_iterations = accumulated_iterations;
    result.mean_iterations_per_step = static_cast<double>(accumulated_iterations) / config_.nt;
    result.max_residual = max_final_residual;
    result.points_per_sec = (static_cast<double>(total_pts) * config_.nt) / result.wallclock_time_sec;
    result.u_final = std::move(u);

    if (verbose) {
        std::cout << "--------------------------------------------------------\n"
                  << "  FDM Solve Completed in " << std::fixed << std::setprecision(4)
                  << result.wallclock_time_sec << " s\n"
                  << "  Total Iterations: " << result.total_iterations
                  << " (Mean: " << std::setprecision(2) << result.mean_iterations_per_step << " it/step)\n"
                  << "  Throughput: " << std::scientific << std::setprecision(3) << result.points_per_sec << " space-time pts/s\n"
                  << "  Max Relaxation Residual: " << std::scientific << std::setprecision(4) << result.max_residual << "\n"
                  << "========================================================\n";
    }

    return result;
}

} // namespace pde_solver
