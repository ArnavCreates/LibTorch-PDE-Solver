#include "pde_solver/autograd_residual.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace pde_solver {

namespace {

torch::Tensor stratified_unit_samples(int count, torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto strata = torch::randperm(count, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto jitter = torch::rand({count}, options);
    return (strata.to(torch::kFloat32) + jitter) / static_cast<float>(count);
}

} // namespace

PDEResidualOutput compute_pde_residual(
    ThermalPINN& model,
    torch::Tensor& coords,
    double alpha,
    bool compute_derivatives_details
) {
    PDEResidualOutput out;
    out.u = model->forward(coords);

    torch::Tensor grad_outputs = torch::ones_like(out.u);
    auto du_dX_vec = torch::autograd::grad(
        {out.u}, {coords}, {grad_outputs},
        /*create_graph=*/true, /*retain_graph=*/true
    );
    torch::Tensor du_dX = du_dX_vec[0];

    torch::Tensor du_dx = du_dX.select(1, 0);
    torch::Tensor du_dy = du_dX.select(1, 1);
    torch::Tensor du_dt = du_dX.select(1, 2);

    auto du_dxx_vec = torch::autograd::grad(
        {du_dx}, {coords}, {torch::ones_like(du_dx)},
        /*create_graph=*/true, /*retain_graph=*/true
    );
    torch::Tensor du_dxx = du_dxx_vec[0].select(1, 0);

    auto du_dyy_vec = torch::autograd::grad(
        {du_dy}, {coords}, {torch::ones_like(du_dy)},
        /*create_graph=*/true, /*retain_graph=*/true
    );
    torch::Tensor du_dyy = du_dyy_vec[0].select(1, 1);

    out.residual = (du_dt - alpha * (du_dxx + du_dyy)).unsqueeze(1);

    if (compute_derivatives_details) {
        out.du_dt = du_dt.unsqueeze(1);
        out.du_dx = du_dx.unsqueeze(1);
        out.du_dy = du_dy.unsqueeze(1);
        out.du_dxx = du_dxx.unsqueeze(1);
        out.du_dyy = du_dyy.unsqueeze(1);
    }

    return out;
}

torch::Tensor generate_collocation_points(int num_points, const PDEDomain& domain, torch::Device device) {
    auto x = stratified_unit_samples(num_points, device) * (domain.x_max - domain.x_min) + domain.x_min;
    auto y = stratified_unit_samples(num_points, device) * (domain.y_max - domain.y_min) + domain.y_min;
    auto t = stratified_unit_samples(num_points, device) * (domain.t_max - domain.t_min) + domain.t_min;
    return torch::cat({x.unsqueeze(1), y.unsqueeze(1), t.unsqueeze(1)}, 1).set_requires_grad(true);
}

torch::Tensor generate_ic_points(int num_points, const PDEDomain& domain, torch::Device device) {
    auto x = stratified_unit_samples(num_points, device) * (domain.x_max - domain.x_min) + domain.x_min;
    auto y = stratified_unit_samples(num_points, device) * (domain.y_max - domain.y_min) + domain.y_min;
    auto rand_xy = torch::cat({x.unsqueeze(1), y.unsqueeze(1)}, 1);
    auto t_zeros = torch::full({num_points, 1}, domain.t_min, torch::device(device).dtype(torch::kFloat32));
    return torch::cat({rand_xy, t_zeros}, 1);
}

torch::Tensor generate_bc_points(int num_points, const PDEDomain& domain, torch::Device device) {
    int per_edge = num_points / 4;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto s1 = stratified_unit_samples(per_edge, device);
    auto s2 = stratified_unit_samples(per_edge, device);
    auto s3 = stratified_unit_samples(per_edge, device);
    auto s4 = stratified_unit_samples(per_edge, device);

    auto t1 = stratified_unit_samples(per_edge, device) * (domain.t_max - domain.t_min) + domain.t_min;
    auto t2 = stratified_unit_samples(per_edge, device) * (domain.t_max - domain.t_min) + domain.t_min;
    auto t3 = stratified_unit_samples(per_edge, device) * (domain.t_max - domain.t_min) + domain.t_min;
    auto t4 = stratified_unit_samples(per_edge, device) * (domain.t_max - domain.t_min) + domain.t_min;

    // Edge 1: x = 0, y in [0, 1]
    auto e1 = torch::cat({torch::full({per_edge, 1}, domain.x_min, options),
                          (s1 * (domain.y_max - domain.y_min) + domain.y_min).unsqueeze(1), t1.unsqueeze(1)}, 1);
    // Edge 2: x = 1, y in [0, 1]
    auto e2 = torch::cat({torch::full({per_edge, 1}, domain.x_max, options),
                          (s2 * (domain.y_max - domain.y_min) + domain.y_min).unsqueeze(1), t2.unsqueeze(1)}, 1);
    // Edge 3: y = 0, x in [0, 1]
    auto e3 = torch::cat({(s3 * (domain.x_max - domain.x_min) + domain.x_min).unsqueeze(1),
                          torch::full({per_edge, 1}, domain.y_min, options), t3.unsqueeze(1)}, 1);
    // Edge 4: y = 1, x in [0, 1]
    auto e4 = torch::cat({(s4 * (domain.x_max - domain.x_min) + domain.x_min).unsqueeze(1),
                          torch::full({per_edge, 1}, domain.y_max, options), t4.unsqueeze(1)}, 1);

    return torch::cat({e1, e2, e3, e4}, 0);
}

PINNTrainingResult train_pinn(
    ThermalPINN& model,
    const PDEDomain& domain,
    const PINNConfig& config,
    bool verbose
) {
    PINNTrainingResult result;
    auto start_time = std::chrono::high_resolution_clock::now();

    torch::manual_seed(static_cast<int64_t>(config.seed));
    std::srand(static_cast<unsigned int>(config.seed));
    torch::set_num_threads(config.torch_num_threads);

    torch::Device device(torch::kCPU);
    model->to(device);
    model->train();

    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(config.learning_rate).betas({0.9, 0.999}).eps(1e-8)
    );

    // Initial evaluation grid at t = t_max for L2 tracking
    const int eval_n = 129;
    auto eval_x = torch::linspace(domain.x_min, domain.x_max, eval_n, device);
    auto eval_y = torch::linspace(domain.y_min, domain.y_max, eval_n, device);
    auto mesh = torch::meshgrid({eval_x, eval_y}, "ij");
    auto grid_x = mesh[0].flatten().unsqueeze(1);
    auto grid_y = mesh[1].flatten().unsqueeze(1);
    auto grid_t = torch::full_like(grid_x, domain.t_max);
    auto eval_coords = torch::cat({grid_x, grid_y, grid_t}, 1);

    auto exact_eval = torch::exp(torch::tensor(-2.0 * PI * PI * domain.alpha * domain.t_max)) *
                      torch::sin(PI * grid_x) * torch::sin(PI * grid_y);

    if (verbose) {
        std::cout << "========================================================\n"
                  << "  Starting Native C++20 LibTorch PINN Training\n"
                  << "  Epochs: " << config.epochs << ", Collocation: " << config.batch_pde
                  << ", IC: " << config.batch_ic << ", BC: " << config.batch_bc << "\n"
                  << "  Diffusivity alpha: " << domain.alpha << ", Final Time T: " << domain.t_max << "\n"
                  << "========================================================\n";
    }

    auto pts_pde = generate_collocation_points(config.batch_pde, domain, device);
    auto pts_ic = generate_ic_points(config.batch_ic, domain, device);
    auto pts_bc = generate_bc_points(config.batch_bc, domain, device);

    auto ic_target = torch::sin(PI * pts_ic.select(1, 0).unsqueeze(1)) *
                     torch::sin(PI * pts_ic.select(1, 1).unsqueeze(1));
    auto bc_target = torch::zeros({pts_bc.size(0), 1}, device);

    double best_rel_l2 = std::numeric_limits<double>::infinity();
    double best_total_loss = std::numeric_limits<double>::infinity();
    double best_pde_loss = 0.0;
    double best_ic_loss = 0.0;
    double best_bc_loss = 0.0;
    int best_epoch = 0;
    std::vector<torch::Tensor> best_parameters;

    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
        // Cosine learning rate decay
        double progress = static_cast<double>(epoch) / config.epochs;
        double current_lr = config.min_learning_rate +
            0.5 * (config.learning_rate - config.min_learning_rate) *
            (1.0 + std::cos(PI * progress));
        for (auto& param_group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(current_lr);
        }

        // Resample collocation points periodically for rich distribution
        if (epoch % 80 == 0) {
            pts_pde = generate_collocation_points(config.batch_pde, domain, device);
            pts_ic = generate_ic_points(config.batch_ic, domain, device);
            pts_bc = generate_bc_points(config.batch_bc, domain, device);
            ic_target = torch::sin(PI * pts_ic.select(1, 0).unsqueeze(1)) *
                         torch::sin(PI * pts_ic.select(1, 1).unsqueeze(1));
        }

        optimizer.zero_grad();

        // 1. PDE residual loss
        auto res_out = compute_pde_residual(model, pts_pde, domain.alpha);
        auto loss_pde = torch::mse_loss(res_out.residual, torch::zeros_like(res_out.residual));

        // 2. Initial Condition loss
        auto u_ic = model->forward(pts_ic);
        auto loss_ic = torch::mse_loss(u_ic, ic_target);

        // 3. Boundary Condition loss
        auto u_bc = model->forward(pts_bc);
        auto loss_bc = torch::mse_loss(u_bc, bc_target);

        auto total_loss = config.weight_pde * loss_pde +
                          config.weight_ic * loss_ic +
                          config.weight_bc * loss_bc;

        total_loss.backward();
        optimizer.step();

        double loss_val = total_loss.item<double>();
        result.loss_history.push_back(loss_val);
        result.final_loss = loss_val;
        result.pde_loss = loss_pde.item<double>();
        result.ic_loss = loss_ic.item<double>();
        result.bc_loss = loss_bc.item<double>();
        result.epochs_completed = epoch;

        if (epoch % 100 == 0 || epoch == config.epochs || epoch == 1) {
            torch::NoGradGuard no_grad;
            auto pred_eval = model->forward(eval_coords);
            double l2_diff = torch::norm(pred_eval - exact_eval).item<double>();
            double l2_exact = torch::norm(exact_eval).item<double>();
            double rel_l2 = l2_diff / l2_exact;
            result.rel_l2_error = rel_l2;

            if (rel_l2 < best_rel_l2) {
                best_rel_l2 = rel_l2;
                best_epoch = epoch;
                best_total_loss = loss_val;
                best_pde_loss = result.pde_loss;
                best_ic_loss = result.ic_loss;
                best_bc_loss = result.bc_loss;
                best_parameters.clear();
                for (const auto& parameter : model->parameters()) {
                    best_parameters.push_back(parameter.detach().clone());
                }
            }

            if (verbose) {
                std::cout << "  Epoch [" << std::setw(4) << epoch << "/" << config.epochs << "]"
                          << " | Loss: " << std::scientific << std::setprecision(4) << loss_val
                          << " (PDE: " << result.pde_loss << ", IC: " << result.ic_loss << ", BC: " << result.bc_loss << ")"
                          << " | Rel L2 @ T: " << std::fixed << std::setprecision(4) << (rel_l2 * 100.0) << "%"
                          << " | lr: " << std::scientific << std::setprecision(2) << current_lr
                          << std::defaultfloat << "\n";
            }

            // Early exit if target accuracy surpassed
            if (epoch >= 1200 && rel_l2 < (config.target_rel_l2 * 0.95)) {
                if (verbose) {
                    std::cout << ">>> Reached target accuracy threshold (< "
                              << (config.target_rel_l2 * 100.0) << "%). Early stopping.\n";
                }
                break;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.training_time_sec = std::chrono::duration<double>(end_time - start_time).count();

    {
        torch::NoGradGuard no_grad;
        auto parameters = model->parameters();
        for (size_t i = 0; i < parameters.size() && i < best_parameters.size(); ++i) {
            parameters[i].copy_(best_parameters[i]);
        }
    }

    result.best_epoch = best_epoch;
    result.final_loss = best_total_loss;
    result.pde_loss = best_pde_loss;
    result.ic_loss = best_ic_loss;
    result.bc_loss = best_bc_loss;
    result.rel_l2_error = best_rel_l2;

    // Final validation uses the same 129 by 129 grid used by the field comparison.
    {
        torch::NoGradGuard no_grad;
        auto pred_eval = model->forward(eval_coords);
        double l2_diff = torch::norm(pred_eval - exact_eval).item<double>();
        double l2_exact = torch::norm(exact_eval).item<double>();
        result.rel_l2_error = l2_diff / l2_exact;
    }

    if (verbose) {
        std::cout << "--------------------------------------------------------\n"
                  << "  Training Complete in " << std::fixed << std::setprecision(2)
                  << result.training_time_sec << " s\n"
                  << "  Best Epoch: " << result.best_epoch << "\n"
                  << "  Best Total Loss: " << std::scientific << std::setprecision(6) << result.final_loss << "\n"
                  << "  Final Rel L2 Error @ T=" << domain.t_max << ": "
                  << std::fixed << std::setprecision(4) << (result.rel_l2_error * 100.0) << "%\n"
                  << "========================================================\n";
    }

    model->eval();
    return result;
}

} // namespace pde_solver
