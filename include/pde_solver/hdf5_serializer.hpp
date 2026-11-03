#ifndef PDE_SOLVER_HDF5_SERIALIZER_HPP
#define PDE_SOLVER_HDF5_SERIALIZER_HPP

#include "pde_solver/config.hpp"
#include "pde_solver/conservation_validator.hpp"
#include <string>
#include <vector>

namespace pde_solver {

struct ThermalFieldsPayload {
    std::vector<double> x_coords;              // Size nx
    std::vector<double> y_coords;              // Size ny
    std::vector<double> u_pinn;                // Size nx * ny
    std::vector<double> u_fdm_gauss_seidel;    // Size nx * ny
    std::vector<double> u_exact;               // Size nx * ny
    std::vector<double> abs_error_pinn_vs_fdm; // Size nx * ny
    std::vector<double> pde_residual_map;      // Size nx * ny

    double alpha = 0.05;
    double final_time = 0.5;
    double relative_l2_error = 0.0;
    double boundary_rmse = 0.0;
    double energy_conservation_rel_error = 0.0;
};

class HDF5Serializer {
public:
    static bool serialize_fields(
        const std::string& filepath,
        const ThermalFieldsPayload& payload,
        bool verbose = true
    );
};

} // namespace pde_solver

#endif // PDE_SOLVER_HDF5_SERIALIZER_HPP
