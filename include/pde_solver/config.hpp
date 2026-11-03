#ifndef PDE_SOLVER_CONFIG_HPP
#define PDE_SOLVER_CONFIG_HPP

#include <vector>
#include <string>
#include <cmath>
#include <numbers>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <utility>

namespace pde_solver {

constexpr double PI = std::numbers::pi;

struct PDEDomain {
    double alpha = 0.05;
    double x_min = 0.0;
    double x_max = 1.0;
    double y_min = 0.0;
    double y_max = 1.0;
    double t_min = 0.0;
    double t_max = 0.5;

    [[nodiscard]] double exact_solution(double x, double y, double t) const noexcept {
        return std::exp(-2.0 * PI * PI * alpha * t) * std::sin(PI * x) * std::sin(PI * y);
    }

    [[nodiscard]] double exact_energy(double t) const noexcept {
        return (4.0 / (PI * PI)) * std::exp(-2.0 * PI * PI * alpha * t);
    }
};

struct FDMConfig {
    int nx = 129;
    int ny = 129;
    int nt = 200;
    int num_threads = 8;
    double tolerance = 1.0e-9;
    int max_iterations = 500;
};

struct PINNConfig {
    int input_dim = 3;
    int output_dim = 1;
    int hidden_dim = 64;
    int num_hidden_layers = 4;
    double learning_rate = 0.003;
    double min_learning_rate = 0.0001;
    int epochs = 1200;
    int batch_pde = 2048;
    int batch_ic = 512;
    int batch_bc = 512;
    double weight_pde = 1.0;
    double weight_ic = 10.0;
    double weight_bc = 10.0;
    double target_rel_l2 = 0.013;
    std::uint64_t seed = 42;
    int torch_num_threads = 1;
};

struct ProfilerConfig {
    std::vector<int> batch_sizes = {256, 1024, 4096, 16384, 65536};
    int warmup_passes = 5;
    int timed_passes = 25;
    double ridge_point_flops_per_byte = 13.0;
};

struct SolverConfig {
    PDEDomain domain;
    FDMConfig fdm;
    PINNConfig pinn;
    ProfilerConfig profiler;
    std::string hdf5_path = "data/thermal_fields_2d.h5";
    std::string benchmark_json = "assets/benchmark_summary.json";
    std::string roofline_json = "assets/roofline_arithmetic_intensity.json";
};

namespace config_detail {

inline std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

inline std::string unquote(std::string value) {
    value = trim(std::move(value));
    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                              (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

inline std::vector<int> parse_int_list(std::string value) {
    value = trim(std::move(value));
    if (value.size() >= 2 && value.front() == '[' && value.back() == ']') {
        value = value.substr(1, value.size() - 2);
    }
    std::vector<int> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim(std::move(item));
        if (!item.empty()) result.push_back(std::stoi(item));
    }
    return result;
}

inline void assign(SolverConfig& config, const std::string& key, const std::string& value) {
    if (key == "seed" || key == "pinn.seed") config.pinn.seed = std::stoull(value);
    else if (key == "pde.alpha") config.domain.alpha = std::stod(value);
    else if (key == "pde.x_min") config.domain.x_min = std::stod(value);
    else if (key == "pde.x_max") config.domain.x_max = std::stod(value);
    else if (key == "pde.y_min") config.domain.y_min = std::stod(value);
    else if (key == "pde.y_max") config.domain.y_max = std::stod(value);
    else if (key == "pde.t_min") config.domain.t_min = std::stod(value);
    else if (key == "pde.t_max" || key == "pde.t_final") config.domain.t_max = std::stod(value);
    else if (key == "fdm.grid_size") config.fdm.nx = config.fdm.ny = std::stoi(value);
    else if (key == "fdm.nx") config.fdm.nx = std::stoi(value);
    else if (key == "fdm.ny") config.fdm.ny = std::stoi(value);
    else if (key == "fdm.nt" || key == "fdm.time_steps") config.fdm.nt = std::stoi(value);
    else if (key == "fdm.threads" || key == "fdm.num_threads") config.fdm.num_threads = std::stoi(value);
    else if (key == "fdm.tolerance") config.fdm.tolerance = std::stod(value);
    else if (key == "fdm.max_iterations") config.fdm.max_iterations = std::stoi(value);
    else if (key == "pinn.input_dim") config.pinn.input_dim = std::stoi(value);
    else if (key == "pinn.output_dim") config.pinn.output_dim = std::stoi(value);
    else if (key == "pinn.hidden_dim") config.pinn.hidden_dim = std::stoi(value);
    else if (key == "pinn.num_hidden_layers") config.pinn.num_hidden_layers = std::stoi(value);
    else if (key == "pinn.learning_rate") config.pinn.learning_rate = std::stod(value);
    else if (key == "pinn.min_learning_rate") config.pinn.min_learning_rate = std::stod(value);
    else if (key == "pinn.epochs") config.pinn.epochs = std::stoi(value);
    else if (key == "pinn.batch_pde" || key == "pinn.collocation_points") config.pinn.batch_pde = std::stoi(value);
    else if (key == "pinn.batch_ic" || key == "pinn.ic_points") config.pinn.batch_ic = std::stoi(value);
    else if (key == "pinn.batch_bc" || key == "pinn.bc_points") config.pinn.batch_bc = std::stoi(value);
    else if (key == "pinn.weight_pde") config.pinn.weight_pde = std::stod(value);
    else if (key == "pinn.weight_ic") config.pinn.weight_ic = std::stod(value);
    else if (key == "pinn.weight_bc") config.pinn.weight_bc = std::stod(value);
    else if (key == "pinn.target_rel_l2") config.pinn.target_rel_l2 = std::stod(value);
    else if (key == "pinn.torch_num_threads") config.pinn.torch_num_threads = std::stoi(value);
    else if (key == "profiler.batch_sizes") config.profiler.batch_sizes = parse_int_list(value);
    else if (key == "profiler.warmup_passes") config.profiler.warmup_passes = std::stoi(value);
    else if (key == "profiler.timed_passes") config.profiler.timed_passes = std::stoi(value);
    else if (key == "profiler.ridge_point_flops_per_byte") config.profiler.ridge_point_flops_per_byte = std::stod(value);
    else if (key == "output.hdf5_path") config.hdf5_path = unquote(value);
    else if (key == "output.benchmark_json") config.benchmark_json = unquote(value);
    else if (key == "output.roofline_json") config.roofline_json = unquote(value);
}

} // namespace config_detail

inline SolverConfig load_solver_config(const std::string& requested_path = {}) {
    std::filesystem::path path;
    if (!requested_path.empty()) {
        path = requested_path;
    } else if (std::filesystem::exists("configs/solver_config.yaml")) {
        path = "configs/solver_config.yaml";
    } else if (std::filesystem::exists("../configs/solver_config.yaml")) {
        path = "../configs/solver_config.yaml";
    } else {
        throw std::runtime_error("Could not locate configs/solver_config.yaml; pass --config <path>.");
    }

    std::ifstream input(path);
    if (!input) throw std::runtime_error("Could not open solver configuration: " + path.string());

    SolverConfig config;
    std::string section;
    std::string line;
    while (std::getline(input, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        if (config_detail::trim(line).empty()) continue;
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = config_detail::trim(line.substr(0, colon));
        const std::string value = config_detail::trim(line.substr(colon + 1));
        if (value.empty()) {
            section = key;
            continue;
        }
        const std::string qualified_key = section.empty() ? key : section + "." + key;
        config_detail::assign(config, qualified_key, value);
    }

    if (config.domain.alpha <= 0.0 || config.domain.t_max <= config.domain.t_min ||
        config.fdm.nx < 3 || config.fdm.ny < 3 || config.fdm.nt <= 0 || config.fdm.num_threads <= 0 ||
        config.pinn.epochs <= 0 || config.pinn.batch_pde <= 0 || config.pinn.seed == 0 ||
        config.profiler.batch_sizes.empty() || config.profiler.timed_passes <= 0) {
        throw std::runtime_error("Solver configuration contains invalid parameter values.");
    }
    return config;
}

} // namespace pde_solver

#endif // PDE_SOLVER_CONFIG_HPP
