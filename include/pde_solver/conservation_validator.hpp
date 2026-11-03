#ifndef PDE_SOLVER_CONSERVATION_VALIDATOR_HPP
#define PDE_SOLVER_CONSERVATION_VALIDATOR_HPP

#include "pde_solver/config.hpp"
#include <vector>

namespace pde_solver {

struct ValidationReport {
    // Relative L2 Errors
    double rel_l2_pinn_vs_exact = 0.0;
    double rel_l2_fdm_vs_exact = 0.0;
    double rel_l2_pinn_vs_fdm = 0.0;

    // L-infinity (Max absolute) Errors
    double max_err_pinn_vs_exact = 0.0;
    double max_err_fdm_vs_exact = 0.0;
    double max_err_pinn_vs_fdm = 0.0;

    // Boundary Condition Fidelity
    double boundary_max_error = 0.0;
    double boundary_rmse = 0.0;

    // Energy Conservation
    double exact_energy_t0 = 0.0;
    double exact_energy_t_final = 0.0;
    double fdm_energy_t_final = 0.0;
    double pinn_energy_t_final = 0.0;
    double fdm_energy_rel_error = 0.0;
    double pinn_energy_rel_error = 0.0;

    // Physical conservation status (< 1% discrepancy)
    bool energy_conserved = false;
    bool l2_criterion_passed = false;
};

class ConservationValidator {
public:
    explicit ConservationValidator(PDEDomain domain);

    ValidationReport validate(
        const std::vector<double>& u_pinn,
        const std::vector<double>& u_fdm,
        const std::vector<double>& u_exact,
        const std::vector<double>& x_coords,
        const std::vector<double>& y_coords,
        bool verbose = true
    ) const;

    // 2D numerical integration over uniform rectangular grid via composite Simpson/trapezoidal rule
    static double integrate_2d(
        const std::vector<double>& grid_data,
        const std::vector<double>& x_coords,
        const std::vector<double>& y_coords
    );

private:
    PDEDomain domain_;
};

} // namespace pde_solver

#endif // PDE_SOLVER_CONSERVATION_VALIDATOR_HPP
