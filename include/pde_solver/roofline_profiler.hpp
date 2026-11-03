#ifndef PDE_SOLVER_ROOFLINE_PROFILER_HPP
#define PDE_SOLVER_ROOFLINE_PROFILER_HPP

#include "pde_solver/pinn_network.hpp"
#include "pde_solver/config.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace pde_solver {

struct BatchRooflinePoint {
    int batch_size = 0;
    double mean_latency_ms = 0.0;
    double min_latency_ms = 0.0;
    double max_latency_ms = 0.0;
    double throughput_pts_per_sec = 0.0;
    double arithmetic_intensity_flops_per_byte = 0.0;
    double effective_gflops_per_sec = 0.0;
    double effective_memory_bandwidth_gb_per_sec = 0.0;
    std::string regime; // "Memory-Bandwidth Bound" or "Compute-Bound"
};

struct RooflineProfileReport {
    std::string device_name = "CPU";
    int num_threads = 1;
    int64_t model_parameter_count = 0;
    double model_parameter_bytes = 0.0;
    double forward_flops_per_point = 0.0;
    double residual_flops_per_point = 0.0;
    double ridge_point_flops_per_byte = 13.0;
    std::vector<BatchRooflinePoint> forward_sweep;
    std::vector<BatchRooflinePoint> residual_sweep;

    [[nodiscard]] std::string to_json() const;
    void save_json(const std::string& filepath) const;
};

class RooflineProfiler {
public:
    explicit RooflineProfiler(ProfilerConfig config);

    RooflineProfileReport run_sweep(ThermalPINN& model, const PDEDomain& domain, bool verbose = true) const;

private:
    ProfilerConfig config_;
};

} // namespace pde_solver

#endif // PDE_SOLVER_ROOFLINE_PROFILER_HPP
