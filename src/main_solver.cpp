#include "pde_solver/config.hpp"
#include "pde_solver/pinn_network.hpp"
#include "pde_solver/autograd_residual.hpp"
#include "pde_solver/fdm_gauss_seidel.hpp"
#include "pde_solver/conservation_validator.hpp"
#include "pde_solver/roofline_profiler.hpp"
#include "pde_solver/hdf5_serializer.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    std::string config_path;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else {
            std::cerr << "Usage: pde_solver_main [--config <path>]\n";
            return EXIT_FAILURE;
        }
    }

    pde_solver::SolverConfig solver_cfg;
    try {
        solver_cfg = pde_solver::load_solver_config(config_path);
    } catch (const std::exception& error) {
        std::cerr << "Configuration error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "LibTorch PDE solve | 2D heat diffusion on "
              << solver_cfg.fdm.nx << " x " << solver_cfg.fdm.ny << " grid\n\n";

    const auto& domain = solver_cfg.domain;
    const auto& fdm_cfg = solver_cfg.fdm;
    const auto& pinn_cfg = solver_cfg.pinn;
    const auto& profiler_cfg = solver_cfg.profiler;

    for (const auto& path : {solver_cfg.hdf5_path, solver_cfg.benchmark_json, solver_cfg.roofline_json}) {
        const auto parent = std::filesystem::path(path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
    }

    std::srand(static_cast<unsigned int>(pinn_cfg.seed));
    torch::manual_seed(static_cast<int64_t>(pinn_cfg.seed));
    torch::set_num_threads(pinn_cfg.torch_num_threads);

    // 1. Train Native LibTorch C++20 PINN Surrogate
    auto pinn = pde_solver::ThermalPINN(
        pinn_cfg.input_dim, pinn_cfg.output_dim, pinn_cfg.hidden_dim,
        pinn_cfg.num_hidden_layers, pinn_cfg.seed
    );
    auto train_result = pde_solver::train_pinn(pinn, domain, pinn_cfg, /*verbose=*/true);

    // 2. Solve 2D Heat Diffusion PDE with OpenMP Red-Black Gauss-Seidel FDM
    torch::set_num_threads(fdm_cfg.num_threads);
    pde_solver::OpenMPRedBlackGaussSeidelSolver fdm_solver(domain, fdm_cfg);
    auto fdm_result = fdm_solver.solve(/*verbose=*/true);
    torch::set_num_threads(pinn_cfg.torch_num_threads);

    // 3. Evaluate PINN Surrogate & Compute Analytical Reference on 129x129 Grid at t = T
    const int nx = fdm_cfg.nx;
    const int ny = fdm_cfg.ny;
    const int total_pts = nx * ny;

    std::vector<double> u_pinn(total_pts, 0.0);
    std::vector<double> u_exact(total_pts, 0.0);
    std::vector<double> abs_err(total_pts, 0.0);
    std::vector<double> residual_map(total_pts, 0.0);

    // Prepare grid tensor for LibTorch evaluation
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto grid_tensor = torch::empty({total_pts, 3}, options);
    auto grid_accessor = grid_tensor.accessor<float, 2>();

    for (int i = 0; i < nx; ++i) {
        float x = static_cast<float>(fdm_result.x_coords[i]);
        for (int j = 0; j < ny; ++j) {
            float y = static_cast<float>(fdm_result.y_coords[j]);
            int idx = i * ny + j;
            grid_accessor[idx][0] = x;
            grid_accessor[idx][1] = y;
            grid_accessor[idx][2] = static_cast<float>(domain.t_max);

            u_exact[idx] = domain.exact_solution(x, y, domain.t_max);
        }
    }

    // Benchmark single-pass PINN forward surrogate evaluation on entire grid
    auto t_surrogate_start = std::chrono::high_resolution_clock::now();
    {
        torch::NoGradGuard no_grad;
        auto pred_tensor = pinn->forward(grid_tensor);
        auto pred_flat = pred_tensor.flatten();
        auto pred_acc = pred_flat.accessor<float, 1>();
        for (int k = 0; k < total_pts; ++k) {
            u_pinn[k] = static_cast<double>(pred_acc[k]);
            abs_err[k] = std::abs(u_pinn[k] - fdm_result.u_final[k]);
        }
    }
    auto t_surrogate_end = std::chrono::high_resolution_clock::now();
    double pinn_grid_eval_time_ms = std::chrono::duration<double, std::milli>(t_surrogate_end - t_surrogate_start).count();
    double pinn_throughput_pts_per_sec = (static_cast<double>(total_pts) / (pinn_grid_eval_time_ms / 1000.0));

    // Compute PDE residual map on grid in chunks of 2048 to prevent memory spikes
    {
        int chunk_size = 2048;
        for (int c = 0; c < total_pts; c += chunk_size) {
            int cur_chunk = std::min(chunk_size, total_pts - c);
            auto slice_tensor = grid_tensor.slice(0, c, c + cur_chunk).clone().detach().set_requires_grad(true);
            auto res_out = pde_solver::compute_pde_residual(pinn, slice_tensor, domain.alpha);
            auto res_flat = res_out.residual.flatten();
            auto res_acc = res_flat.accessor<float, 1>();
            for (int k = 0; k < cur_chunk; ++k) {
                residual_map[c + k] = static_cast<double>(res_acc[k]);
            }
        }
    }

    // 4. Conservation & Physical Accuracy Validation
    pde_solver::ConservationValidator validator(domain);
    auto val_report = validator.validate(
        u_pinn, fdm_result.u_final, u_exact,
        fdm_result.x_coords, fdm_result.y_coords,
        /*verbose=*/true
    );

    // 5. Arithmetic Intensity & Roofline Batch Sweep Profiling
    pde_solver::RooflineProfiler profiler(profiler_cfg);
    auto roofline_report = profiler.run_sweep(pinn, domain, /*verbose=*/true);
    roofline_report.save_json(solver_cfg.roofline_json);

    // 6. Serialize Full 2D Spatial Fields to Native HDF5
    pde_solver::ThermalFieldsPayload hdf5_payload;
    hdf5_payload.x_coords = fdm_result.x_coords;
    hdf5_payload.y_coords = fdm_result.y_coords;
    hdf5_payload.u_pinn = u_pinn;
    hdf5_payload.u_fdm_gauss_seidel = fdm_result.u_final;
    hdf5_payload.u_exact = u_exact;
    hdf5_payload.abs_error_pinn_vs_fdm = abs_err;
    hdf5_payload.pde_residual_map = residual_map;
    hdf5_payload.alpha = domain.alpha;
    hdf5_payload.final_time = domain.t_max;
    hdf5_payload.relative_l2_error = val_report.rel_l2_pinn_vs_exact;
    hdf5_payload.boundary_rmse = val_report.boundary_rmse;
    hdf5_payload.energy_conservation_rel_error = val_report.pinn_energy_rel_error;

    const bool hdf5_saved = pde_solver::HDF5Serializer::serialize_fields(
        solver_cfg.hdf5_path, hdf5_payload, /*verbose=*/true);
    if (!hdf5_saved) return EXIT_FAILURE;

    // 7. Export Comprehensive Benchmark Summary JSON
    double fdm_single_step_ms = (fdm_result.wallclock_time_sec * 1000.0) / fdm_cfg.nt;
    double surrogate_speedup = (fdm_result.wallclock_time_sec * 1000.0) / pinn_grid_eval_time_ms;

    std::ostringstream json_ss;
    json_ss << std::fixed << std::setprecision(6)
            << "{\n"
            << "  \"metadata\": {\n"
            << "    \"equation\": \"2D Transient Heat Diffusion Equation\",\n"
            << "    \"alpha\": " << domain.alpha << ",\n"
            << "    \"domain\": \"[" << domain.x_min << ", " << domain.x_max << "] x ["
            << domain.y_min << ", " << domain.y_max << "] x [" << domain.t_min << ", " << domain.t_max << "]\",\n"
            << "    \"grid_resolution\": [" << nx << ", " << ny << "],\n"
            << "    \"time_steps\": " << fdm_cfg.nt << "\n"
            << "  },\n"
            << "  \"pinn_training\": {\n"
            << "    \"epochs\": " << train_result.epochs_completed << ",\n"
            << "    \"best_epoch\": " << train_result.best_epoch << ",\n"
            << "    \"seed\": " << pinn_cfg.seed << ",\n"
            << "    \"total_loss\": " << train_result.final_loss << ",\n"
            << "    \"pde_loss\": " << train_result.pde_loss << ",\n"
            << "    \"ic_loss\": " << train_result.ic_loss << ",\n"
            << "    \"bc_loss\": " << train_result.bc_loss << ",\n"
            << "    \"training_time_sec\": " << train_result.training_time_sec << ",\n"
            << "    \"model_parameters\": " << pinn->count_parameters() << "\n"
            << "  },\n"
            << "  \"fdm_benchmark\": {\n"
            << "    \"wallclock_time_sec\": " << fdm_result.wallclock_time_sec << ",\n"
            << "    \"single_step_time_ms\": " << fdm_single_step_ms << ",\n"
            << "    \"total_iterations\": " << fdm_result.total_iterations << ",\n"
            << "    \"mean_iterations_per_step\": " << fdm_result.mean_iterations_per_step << ",\n"
            << "    \"throughput_pts_per_sec\": " << fdm_result.points_per_sec << ",\n"
            << "    \"threads\": " << fdm_result.num_threads_used << "\n"
            << "  },\n"
            << "  \"surrogate_evaluation\": {\n"
            << "    \"grid_eval_time_ms\": " << pinn_grid_eval_time_ms << ",\n"
            << "    \"throughput_pts_per_sec\": " << pinn_throughput_pts_per_sec << ",\n"
            << "    \"speedup_vs_total_fdm\": " << surrogate_speedup << "\n"
            << "  },\n"
            << "  \"accuracy_and_conservation\": {\n"
            << "    \"rel_l2_pinn_vs_exact\": " << val_report.rel_l2_pinn_vs_exact << ",\n"
            << "    \"rel_l2_fdm_vs_exact\": " << val_report.rel_l2_fdm_vs_exact << ",\n"
            << "    \"rel_l2_pinn_vs_fdm\": " << val_report.rel_l2_pinn_vs_fdm << ",\n"
            << "    \"max_err_pinn_vs_exact\": " << val_report.max_err_pinn_vs_exact << ",\n"
            << "    \"max_err_fdm_vs_exact\": " << val_report.max_err_fdm_vs_exact << ",\n"
            << "    \"boundary_max_error\": " << val_report.boundary_max_error << ",\n"
            << "    \"boundary_rmse\": " << val_report.boundary_rmse << ",\n"
            << "    \"exact_energy_t0\": " << val_report.exact_energy_t0 << ",\n"
            << "    \"exact_energy_t_final\": " << val_report.exact_energy_t_final << ",\n"
            << "    \"fdm_energy_t_final\": " << val_report.fdm_energy_t_final << ",\n"
            << "    \"pinn_energy_t_final\": " << val_report.pinn_energy_t_final << ",\n"
            << "    \"fdm_energy_rel_error\": " << val_report.fdm_energy_rel_error << ",\n"
            << "    \"pinn_energy_rel_error\": " << val_report.pinn_energy_rel_error << "\n"
            << "  }\n"
            << "}\n";

    std::ofstream b_out(solver_cfg.benchmark_json);
    if (b_out.is_open()) {
        b_out << json_ss.str();
        std::cout << ">>> Exported Benchmark Summary: " << solver_cfg.benchmark_json << "\n";
    } else {
        std::cerr << "Could not write benchmark file: " << solver_cfg.benchmark_json << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nSolver run finished.\n";

    return (val_report.l2_criterion_passed && val_report.energy_conserved)
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
