#ifndef PDE_SOLVER_AUTOGRAD_RESIDUAL_HPP
#define PDE_SOLVER_AUTOGRAD_RESIDUAL_HPP

#include "pde_solver/pinn_network.hpp"
#include "pde_solver/config.hpp"
#include <torch/torch.h>
#include <vector>

namespace pde_solver {

struct PDEResidualOutput {
    torch::Tensor u;
    torch::Tensor residual;
    torch::Tensor du_dt;
    torch::Tensor du_dx;
    torch::Tensor du_dy;
    torch::Tensor du_dxx;
    torch::Tensor du_dyy;
};

struct PINNTrainingResult {
    double final_loss = 0.0;
    double pde_loss = 0.0;
    double ic_loss = 0.0;
    double bc_loss = 0.0;
    double rel_l2_error = 0.0;
    double training_time_sec = 0.0;
    int epochs_completed = 0;
    int best_epoch = 0;
    std::vector<double> loss_history;
};

PDEResidualOutput compute_pde_residual(
    ThermalPINN& model,
    torch::Tensor& coords,
    double alpha,
    bool compute_derivatives_details = false
);

torch::Tensor generate_collocation_points(int num_points, const PDEDomain& domain, torch::Device device);
torch::Tensor generate_ic_points(int num_points, const PDEDomain& domain, torch::Device device);
torch::Tensor generate_bc_points(int num_points, const PDEDomain& domain, torch::Device device);

PINNTrainingResult train_pinn(
    ThermalPINN& model,
    const PDEDomain& domain,
    const PINNConfig& config,
    bool verbose = true
);

} // namespace pde_solver

#endif // PDE_SOLVER_AUTOGRAD_RESIDUAL_HPP
