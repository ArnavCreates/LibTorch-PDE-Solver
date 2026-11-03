#include "pde_solver/pinn_network.hpp"
#include <string>

namespace pde_solver {

ThermalPINNImpl::ThermalPINNImpl(int input_dim,
                                 int output_dim,
                                 int hidden_dim,
                                 int num_hidden_layers,
                                 std::uint64_t seed)
    : input_dim_(input_dim),
      output_dim_(output_dim),
      hidden_dim_(hidden_dim),
      num_hidden_layers_(num_hidden_layers) {

    torch::manual_seed(static_cast<int64_t>(seed));

    input_layer_ = register_module("input_layer", torch::nn::Linear(input_dim_, hidden_dim_));

    for (int i = 0; i < num_hidden_layers_ - 1; ++i) {
        hidden_layers_.push_back(
            register_module("hidden_" + std::to_string(i), torch::nn::Linear(hidden_dim_, hidden_dim_))
        );
    }

    output_layer_ = register_module("output_layer", torch::nn::Linear(hidden_dim_, output_dim_));

    reset_parameters();
}

void ThermalPINNImpl::reset_parameters() {
    torch::NoGradGuard no_grad;

    torch::nn::init::xavier_uniform_(input_layer_->weight);
    torch::nn::init::zeros_(input_layer_->bias);

    for (auto& layer : hidden_layers_) {
        torch::nn::init::xavier_uniform_(layer->weight);
        torch::nn::init::zeros_(layer->bias);
    }

    torch::nn::init::xavier_uniform_(output_layer_->weight);
    torch::nn::init::zeros_(output_layer_->bias);
}

torch::Tensor ThermalPINNImpl::forward(torch::Tensor x) {
    x = torch::tanh(input_layer_(x));
    for (auto& layer : hidden_layers_) {
        x = torch::tanh(layer(x));
    }
    x = output_layer_(x);
    return x;
}

int64_t ThermalPINNImpl::count_parameters() const {
    int64_t total = 0;
    for (const auto& p : parameters()) {
        total += p.numel();
    }
    return total;
}

} // namespace pde_solver
