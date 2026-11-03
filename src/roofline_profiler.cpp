#include "pde_solver/roofline_profiler.hpp"
#include "pde_solver/autograd_residual.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace pde_solver {

RooflineProfiler::RooflineProfiler(ProfilerConfig config)
    : config_(std::move(config)) {}

RooflineProfileReport RooflineProfiler::run_sweep(
    ThermalPINN& model,
    const PDEDomain& domain,
    bool verbose
) const {
    RooflineProfileReport report;
    report.device_name = "CPU";
    report.num_threads = torch::get_num_threads();
    report.model_parameter_count = model->count_parameters();
    report.model_parameter_bytes = static_cast<double>(report.model_parameter_count) * sizeof(float);
    report.ridge_point_flops_per_byte = config_.ridge_point_flops_per_byte;

    int in_dim = model->get_input_dim();
    int hidden_dim = model->get_hidden_dim();
    int num_layers = model->get_num_hidden_layers();
    int out_dim = model->get_output_dim();

    // Exact FLOPs per point:
    // Linear layer: 2 * in * out + out (MAC + bias)
    // Tanh activation: ~8 FLOPs per unit
    double layer_flops = 0.0;
    layer_flops += (2.0 * in_dim * hidden_dim + hidden_dim) + (hidden_dim * 8.0);
    for (int l = 0; l < num_layers - 1; ++l) {
        layer_flops += (2.0 * hidden_dim * hidden_dim + hidden_dim) + (hidden_dim * 8.0);
    }
    layer_flops += (2.0 * hidden_dim * out_dim + out_dim);

    report.forward_flops_per_point = layer_flops;
    // 2nd-order PDE autograd requires forward + 1st grad + two 2nd grads ~ 5.5x forward
    report.residual_flops_per_point = layer_flops * 5.5;

    torch::Device device(torch::kCPU);
    model->to(device);
    model->eval();

    if (verbose) {
        std::cout << "========================================================\n"
                  << "  Starting Arithmetic Intensity & Roofline Batch Sweep\n"
                  << "  Threads: " << report.num_threads
                  << ", Parameters: " << report.model_parameter_count
                  << " (" << (report.model_parameter_bytes / 1024.0) << " KiB)\n"
                  << "  Forward FLOPs/pt: " << report.forward_flops_per_point
                  << ", Residual FLOPs/pt: " << report.residual_flops_per_point << "\n"
                  << "========================================================\n";
    }

    // 1. Forward Inference Batch Sweep
    if (verbose) {
        std::cout << ">>> Phase A: Forward Surrogate Evaluation Sweep\n"
                  << "  Batch    Latency(ms)   Throughput(pts/s)   AI(FLOP/B)   GFLOP/s   GB/s    Regime\n"
                  << "  ---------------------------------------------------------------------------------\n";
    }

    for (int B : config_.batch_sizes) {
        auto coords = generate_collocation_points(B, domain, device);
        coords.set_requires_grad(false);

        // Warmup
        for (int w = 0; w < config_.warmup_passes; ++w) {
            torch::NoGradGuard no_grad;
            auto out = model->forward(coords);
        }

        // Timed passes
        std::vector<double> latencies_ms;
        latencies_ms.reserve(config_.timed_passes);

        for (int t = 0; t < config_.timed_passes; ++t) {
            auto t0 = std::chrono::high_resolution_clock::now();
            {
                torch::NoGradGuard no_grad;
                auto out = model->forward(coords);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            latencies_ms.push_back(ms);
        }

        double sum_ms = 0.0;
        double min_ms = latencies_ms.front();
        double max_ms = latencies_ms.front();
        for (double m : latencies_ms) {
            sum_ms += m;
            min_ms = std::min(min_ms, m);
            max_ms = std::max(max_ms, m);
        }
        double mean_ms = sum_ms / config_.timed_passes;
        double mean_sec = mean_ms / 1000.0;

        BatchRooflinePoint pt;
        pt.batch_size = B;
        pt.mean_latency_ms = mean_ms;
        pt.min_latency_ms = min_ms;
        pt.max_latency_ms = max_ms;
        pt.throughput_pts_per_sec = static_cast<double>(B) / mean_sec;

        // Estimated bytes include coordinates, saved forward activations, output, and amortized model weights.
        double bytes_per_point = (in_dim * sizeof(float)) +
                                 (num_layers * hidden_dim * sizeof(float) * 2.0) +
                                 (out_dim * sizeof(float)) +
                                 (report.model_parameter_bytes / B);

        pt.arithmetic_intensity_flops_per_byte = report.forward_flops_per_point / bytes_per_point;
        double total_flops = report.forward_flops_per_point * B;
        pt.effective_gflops_per_sec = (total_flops / mean_sec) / 1.0e9;
        pt.effective_memory_bandwidth_gb_per_sec = (bytes_per_point * B / mean_sec) / 1.0e9;

        pt.regime = (pt.arithmetic_intensity_flops_per_byte < report.ridge_point_flops_per_byte)
                        ? "Memory-Bandwidth Bound"
                        : "Compute-Bound";

        report.forward_sweep.push_back(pt);

        if (verbose) {
            std::cout << "  " << std::setw(6) << B
                      << "   " << std::setw(9) << std::fixed << std::setprecision(3) << pt.mean_latency_ms
                      << "   " << std::setw(15) << std::scientific << std::setprecision(2) << pt.throughput_pts_per_sec
                      << "   " << std::setw(10) << std::fixed << std::setprecision(2) << pt.arithmetic_intensity_flops_per_byte
                      << "   " << std::setw(7) << std::fixed << std::setprecision(2) << pt.effective_gflops_per_sec
                      << "   " << std::setw(6) << std::fixed << std::setprecision(2) << pt.effective_memory_bandwidth_gb_per_sec
                      << "  " << pt.regime << "\n";
        }
    }

    // 2. 2nd-Order PDE Autograd Residual Sweep
    if (verbose) {
        std::cout << "\n>>> Phase B: 2nd-Order Autograd Residual Graph Evaluation Sweep\n"
                  << "  Batch    Latency(ms)   Throughput(pts/s)   AI(FLOP/B)   GFLOP/s   GB/s    Regime\n"
                  << "  ---------------------------------------------------------------------------------\n";
    }

    for (int B : config_.batch_sizes) {
        auto coords = generate_collocation_points(B, domain, device);

        // Warmup
        for (int w = 0; w < config_.warmup_passes; ++w) {
            auto out = compute_pde_residual(model, coords, domain.alpha);
        }

        // Timed passes
        std::vector<double> latencies_ms;
        latencies_ms.reserve(config_.timed_passes);

        for (int t = 0; t < config_.timed_passes; ++t) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto out = compute_pde_residual(model, coords, domain.alpha);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            latencies_ms.push_back(ms);
        }

        double sum_ms = 0.0;
        double min_ms = latencies_ms.front();
        double max_ms = latencies_ms.front();
        for (double m : latencies_ms) {
            sum_ms += m;
            min_ms = std::min(min_ms, m);
            max_ms = std::max(max_ms, m);
        }
        double mean_ms = sum_ms / config_.timed_passes;
        double mean_sec = mean_ms / 1000.0;

        BatchRooflinePoint pt;
        pt.batch_size = B;
        pt.mean_latency_ms = mean_ms;
        pt.min_latency_ms = min_ms;
        pt.max_latency_ms = max_ms;
        pt.throughput_pts_per_sec = static_cast<double>(B) / mean_sec;

        // Estimate saved activations and derivative buffers, plus amortized weight and gradient-sized parameter traffic.
        double bytes_per_point = (in_dim * sizeof(float)) +
                                 (num_layers * hidden_dim * sizeof(float) * 11.25) +
                                 (out_dim * sizeof(float)) +
                                 (2.0 * report.model_parameter_bytes / B);

        pt.arithmetic_intensity_flops_per_byte = report.residual_flops_per_point / bytes_per_point;
        double total_flops = report.residual_flops_per_point * B;
        pt.effective_gflops_per_sec = (total_flops / mean_sec) / 1.0e9;
        pt.effective_memory_bandwidth_gb_per_sec = (bytes_per_point * B / mean_sec) / 1.0e9;

        pt.regime = (pt.arithmetic_intensity_flops_per_byte < report.ridge_point_flops_per_byte)
                        ? "Memory-Bandwidth Bound"
                        : "Compute-Bound";

        report.residual_sweep.push_back(pt);

        if (verbose) {
            std::cout << "  " << std::setw(6) << B
                      << "   " << std::setw(9) << std::fixed << std::setprecision(3) << pt.mean_latency_ms
                      << "   " << std::setw(15) << std::scientific << std::setprecision(2) << pt.throughput_pts_per_sec
                      << "   " << std::setw(10) << std::fixed << std::setprecision(2) << pt.arithmetic_intensity_flops_per_byte
                      << "   " << std::setw(7) << std::fixed << std::setprecision(2) << pt.effective_gflops_per_sec
                      << "   " << std::setw(6) << std::fixed << std::setprecision(2) << pt.effective_memory_bandwidth_gb_per_sec
                      << "  " << pt.regime << "\n";
        }
    }

    if (verbose) {
        std::cout << "========================================================\n"
                  << std::defaultfloat;
    }

    return report;
}

std::string RooflineProfileReport::to_json() const {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"device\": \"" << device_name << "\",\n"
        << "  \"num_threads\": " << num_threads << ",\n"
        << "  \"model_parameter_count\": " << model_parameter_count << ",\n"
        << "  \"model_parameter_bytes\": " << model_parameter_bytes << ",\n"
        << "  \"ridge_point_flops_per_byte\": " << ridge_point_flops_per_byte << ",\n"
        << "  \"forward_flops_per_point\": " << forward_flops_per_point << ",\n"
        << "  \"residual_flops_per_point\": " << residual_flops_per_point << ",\n"
        << "  \"forward_sweep\": [\n";

    for (size_t i = 0; i < forward_sweep.size(); ++i) {
        const auto& pt = forward_sweep[i];
        oss << "    {\n"
            << "      \"batch_size\": " << pt.batch_size << ",\n"
            << "      \"mean_latency_ms\": " << pt.mean_latency_ms << ",\n"
            << "      \"min_latency_ms\": " << pt.min_latency_ms << ",\n"
            << "      \"max_latency_ms\": " << pt.max_latency_ms << ",\n"
            << "      \"throughput_pts_per_sec\": " << pt.throughput_pts_per_sec << ",\n"
            << "      \"arithmetic_intensity_flops_per_byte\": " << pt.arithmetic_intensity_flops_per_byte << ",\n"
            << "      \"effective_gflops_per_sec\": " << pt.effective_gflops_per_sec << ",\n"
            << "      \"effective_memory_bandwidth_gb_per_sec\": " << pt.effective_memory_bandwidth_gb_per_sec << ",\n"
            << "      \"regime\": \"" << pt.regime << "\"\n"
            << "    }" << (i + 1 < forward_sweep.size() ? "," : "") << "\n";
    }

    oss << "  ],\n"
        << "  \"residual_sweep\": [\n";

    for (size_t i = 0; i < residual_sweep.size(); ++i) {
        const auto& pt = residual_sweep[i];
        oss << "    {\n"
            << "      \"batch_size\": " << pt.batch_size << ",\n"
            << "      \"mean_latency_ms\": " << pt.mean_latency_ms << ",\n"
            << "      \"min_latency_ms\": " << pt.min_latency_ms << ",\n"
            << "      \"max_latency_ms\": " << pt.max_latency_ms << ",\n"
            << "      \"throughput_pts_per_sec\": " << pt.throughput_pts_per_sec << ",\n"
            << "      \"arithmetic_intensity_flops_per_byte\": " << pt.arithmetic_intensity_flops_per_byte << ",\n"
            << "      \"effective_gflops_per_sec\": " << pt.effective_gflops_per_sec << ",\n"
            << "      \"effective_memory_bandwidth_gb_per_sec\": " << pt.effective_memory_bandwidth_gb_per_sec << ",\n"
            << "      \"regime\": \"" << pt.regime << "\"\n"
            << "    }" << (i + 1 < residual_sweep.size() ? "," : "") << "\n";
    }

    oss << "  ]\n"
        << "}\n";

    return oss.str();
}

void RooflineProfileReport::save_json(const std::string& filepath) const {
    std::ofstream ofs(filepath);
    if (ofs.is_open()) {
        ofs << to_json();
    }
}

} // namespace pde_solver
