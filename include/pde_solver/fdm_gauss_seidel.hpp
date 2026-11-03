#ifndef PDE_SOLVER_FDM_GAUSS_SEIDEL_HPP
#define PDE_SOLVER_FDM_GAUSS_SEIDEL_HPP

#include "pde_solver/config.hpp"
#include <vector>
#include <cstdint>

namespace pde_solver {

struct FDMResult {
    std::vector<double> u_final;       // Size nx * ny at t = t_max
    std::vector<double> x_coords;      // Size nx
    std::vector<double> y_coords;      // Size ny
    double wallclock_time_sec = 0.0;
    int total_time_steps = 0;
    int64_t total_iterations = 0;
    double mean_iterations_per_step = 0.0;
    double points_per_sec = 0.0;
    double max_residual = 0.0;
    int num_threads_used = 1;
};

class OpenMPRedBlackGaussSeidelSolver {
public:
    OpenMPRedBlackGaussSeidelSolver(PDEDomain domain, FDMConfig config);

    FDMResult solve(bool verbose = true);

    [[nodiscard]] const PDEDomain& get_domain() const noexcept { return domain_; }
    [[nodiscard]] const FDMConfig& get_config() const noexcept { return config_; }

private:
    PDEDomain domain_;
    FDMConfig config_;
};

} // namespace pde_solver

#endif // PDE_SOLVER_FDM_GAUSS_SEIDEL_HPP
