#include "pde_solver/conservation_validator.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace pde_solver {

ConservationValidator::ConservationValidator(PDEDomain domain)
    : domain_(domain) {}

double ConservationValidator::integrate_2d(
    const std::vector<double>& grid_data,
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords
) {
    const size_t nx = x_coords.size();
    const size_t ny = y_coords.size();
    if (grid_data.size() != nx * ny || nx < 3 || ny < 3) {
        return 0.0;
    }

    const double dx = (x_coords.back() - x_coords.front()) / (nx - 1);
    const double dy = (y_coords.back() - y_coords.front()) / (ny - 1);

    // Composite Simpson's 1/3 rule weights for odd number of points
    std::vector<double> wx(nx);
    if (nx % 2 == 1) {
        wx[0] = 1.0;
        wx[nx - 1] = 1.0;
        for (size_t i = 1; i < nx - 1; ++i) {
            wx[i] = (i % 2 == 1) ? 4.0 : 2.0;
        }
    } else {
        // Trapezoidal fallback for even points
        wx[0] = 0.5;
        wx[nx - 1] = 0.5;
        for (size_t i = 1; i < nx - 1; ++i) {
            wx[i] = 1.0;
        }
    }

    std::vector<double> wy(ny);
    if (ny % 2 == 1) {
        wy[0] = 1.0;
        wy[ny - 1] = 1.0;
        for (size_t j = 1; j < ny - 1; ++j) {
            wy[j] = (j % 2 == 1) ? 4.0 : 2.0;
        }
    } else {
        wy[0] = 0.5;
        wy[ny - 1] = 0.5;
        for (size_t j = 1; j < ny - 1; ++j) {
            wy[j] = 1.0;
        }
    }

    double factor = (dx / (nx % 2 == 1 ? 3.0 : 1.0)) * (dy / (ny % 2 == 1 ? 3.0 : 1.0));
    double integral = 0.0;

    for (size_t i = 0; i < nx; ++i) {
        double w_i = wx[i];
        for (size_t j = 0; j < ny; ++j) {
            integral += w_i * wy[j] * grid_data[i * ny + j];
        }
    }

    return integral * factor;
}

ValidationReport ConservationValidator::validate(
    const std::vector<double>& u_pinn,
    const std::vector<double>& u_fdm,
    const std::vector<double>& u_exact,
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords,
    bool verbose
) const {
    ValidationReport rep;
    const size_t nx = x_coords.size();
    const size_t ny = y_coords.size();
    const size_t n_total = nx * ny;

    // 1. Errors against analytical exact solution
    double diff_pinn_exact_sq = 0.0;
    double diff_fdm_exact_sq = 0.0;
    double diff_pinn_fdm_sq = 0.0;
    double norm_exact_sq = 0.0;

    double max_pinn_exact = 0.0;
    double max_fdm_exact = 0.0;
    double max_pinn_fdm = 0.0;

    for (size_t k = 0; k < n_total; ++k) {
        double p = u_pinn[k];
        double f = u_fdm[k];
        double e = u_exact[k];

        norm_exact_sq += e * e;

        double d_pe = std::abs(p - e);
        double d_fe = std::abs(f - e);
        double d_pf = std::abs(p - f);

        diff_pinn_exact_sq += d_pe * d_pe;
        diff_fdm_exact_sq += d_fe * d_fe;
        diff_pinn_fdm_sq += d_pf * d_pf;

        max_pinn_exact = std::max(max_pinn_exact, d_pe);
        max_fdm_exact = std::max(max_fdm_exact, d_fe);
        max_pinn_fdm = std::max(max_pinn_fdm, d_pf);
    }

    double norm_exact = std::sqrt(norm_exact_sq);
    rep.rel_l2_pinn_vs_exact = std::sqrt(diff_pinn_exact_sq) / norm_exact;
    rep.rel_l2_fdm_vs_exact = std::sqrt(diff_fdm_exact_sq) / norm_exact;
    rep.rel_l2_pinn_vs_fdm = std::sqrt(diff_pinn_fdm_sq) / norm_exact;

    rep.max_err_pinn_vs_exact = max_pinn_exact;
    rep.max_err_fdm_vs_exact = max_fdm_exact;
    rep.max_err_pinn_vs_fdm = max_pinn_fdm;

    // 2. Boundary condition fidelity along spatial boundaries
    double bc_sum_sq = 0.0;
    double bc_max = 0.0;
    size_t bc_count = 0;

    for (size_t i = 0; i < nx; ++i) {
        // Bottom boundary: j = 0
        double val_bottom = std::abs(u_pinn[i * ny + 0]);
        bc_sum_sq += val_bottom * val_bottom;
        bc_max = std::max(bc_max, val_bottom);
        bc_count++;

        // Top boundary: j = ny - 1
        double val_top = std::abs(u_pinn[i * ny + (ny - 1)]);
        bc_sum_sq += val_top * val_top;
        bc_max = std::max(bc_max, val_top);
        bc_count++;
    }

    for (size_t j = 1; j < ny - 1; ++j) {
        // Left boundary: i = 0
        double val_left = std::abs(u_pinn[0 * ny + j]);
        bc_sum_sq += val_left * val_left;
        bc_max = std::max(bc_max, val_left);
        bc_count++;

        // Right boundary: i = nx - 1
        double val_right = std::abs(u_pinn[(nx - 1) * ny + j]);
        bc_sum_sq += val_right * val_right;
        bc_max = std::max(bc_max, val_right);
        bc_count++;
    }

    rep.boundary_max_error = bc_max;
    rep.boundary_rmse = std::sqrt(bc_sum_sq / bc_count);

    // 3. Thermal Energy conservation
    rep.exact_energy_t0 = domain_.exact_energy(domain_.t_min);
    rep.exact_energy_t_final = domain_.exact_energy(domain_.t_max);

    rep.fdm_energy_t_final = integrate_2d(u_fdm, x_coords, y_coords);
    rep.pinn_energy_t_final = integrate_2d(u_pinn, x_coords, y_coords);

    rep.fdm_energy_rel_error = std::abs(rep.fdm_energy_t_final - rep.exact_energy_t_final) / rep.exact_energy_t_final;
    rep.pinn_energy_rel_error = std::abs(rep.pinn_energy_t_final - rep.exact_energy_t_final) / rep.exact_energy_t_final;

    rep.energy_conserved = (rep.pinn_energy_rel_error < 0.01) && (rep.fdm_energy_rel_error < 0.01);
    rep.l2_criterion_passed = (rep.rel_l2_pinn_vs_exact < 0.015);

    if (verbose) {
        std::cout << "Numerical results at t = " << domain_.t_max << ":\n"
                  << "  Accuracy Metrics (at t = " << domain_.t_max << "):\n"
                  << "    - Relative L2 Error (PINN vs Exact): "
                  << std::fixed << std::setprecision(4) << (rep.rel_l2_pinn_vs_exact * 100.0) << "%\n"
                  << "    - Relative L2 Error (FDM vs Exact):  "
                  << std::fixed << std::setprecision(4) << (rep.rel_l2_fdm_vs_exact * 100.0) << "%\n"
                  << "    - Relative L2 Error (PINN vs FDM):   "
                  << std::fixed << std::setprecision(4) << (rep.rel_l2_pinn_vs_fdm * 100.0) << "%\n"
                  << "    - Max Pointwise Err (PINN vs Exact): "
                  << std::scientific << std::setprecision(4) << rep.max_err_pinn_vs_exact << "\n"
                  << "    - Max Pointwise Err (FDM vs Exact):  "
                  << std::scientific << std::setprecision(4) << rep.max_err_fdm_vs_exact << "\n\n"
                  << "  Boundary Condition Fidelity (dOmega):\n"
                  << "    - Boundary Max Error: " << std::scientific << std::setprecision(4) << rep.boundary_max_error << "\n"
                  << "    - Boundary RMSE:      " << std::scientific << std::setprecision(4) << rep.boundary_rmse << "\n\n"
                  << "  Thermal Energy Conservation:\n"
                  << "    - Exact Energy @ t=0:     " << std::fixed << std::setprecision(6) << rep.exact_energy_t0 << "\n"
                  << "    - Exact Energy @ t=" << domain_.t_max << ": " << rep.exact_energy_t_final << "\n"
                  << "    - FDM Energy @ t=" << domain_.t_max << ":   " << rep.fdm_energy_t_final
                  << " (rel err: " << std::setprecision(4) << (rep.fdm_energy_rel_error * 100.0) << "%)\n"
                  << "    - PINN Energy @ t=" << domain_.t_max << ":  " << rep.pinn_energy_t_final
                  << " (rel err: " << std::setprecision(4) << (rep.pinn_energy_rel_error * 100.0) << "%)\n"
                  << std::defaultfloat;
    }

    return rep;
}

} // namespace pde_solver
