#ifndef PDE_SOLVER_PINN_NETWORK_HPP
#define PDE_SOLVER_PINN_NETWORK_HPP

#include <torch/torch.h>
#include <vector>
#include <cstdint>

namespace pde_solver {

class ThermalPINNImpl : public torch::nn::Module {
public:
    ThermalPINNImpl(int input_dim = 3,
                    int output_dim = 1,
                    int hidden_dim = 64,
                    int num_hidden_layers = 4,
                    std::uint64_t seed = 42);

    torch::Tensor forward(torch::Tensor x);
    void reset_parameters();

    [[nodiscard]] int get_input_dim() const noexcept { return input_dim_; }
    [[nodiscard]] int get_output_dim() const noexcept { return output_dim_; }
    [[nodiscard]] int get_hidden_dim() const noexcept { return hidden_dim_; }
    [[nodiscard]] int get_num_hidden_layers() const noexcept { return num_hidden_layers_; }

    [[nodiscard]] int64_t count_parameters() const;

private:
    int input_dim_;
    int output_dim_;
    int hidden_dim_;
    int num_hidden_layers_;

    torch::nn::Linear input_layer_{nullptr};
    std::vector<torch::nn::Linear> hidden_layers_;
    torch::nn::Linear output_layer_{nullptr};
};

TORCH_MODULE(ThermalPINN);

} // namespace pde_solver

#endif // PDE_SOLVER_PINN_NETWORK_HPP
