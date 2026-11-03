#include <gtest/gtest.h>
#include <torch/torch.h>
#include "pde_solver/config.hpp"
#include "pde_solver/pinn_network.hpp"
#include "pde_solver/autograd_residual.hpp"
#include "pde_solver/fdm_gauss_seidel.hpp"
#include "pde_solver/conservation_validator.hpp"
#include "pde_solver/hdf5_serializer.hpp"

#include <cmath>
#include <vector>

namespace pde_solver::testing {

TEST(AutogradTest, HigherOrderDerivativesMatchExactMath) {
    // Test on u(x, y, t) = x^2 * y + y^3 + sin(t)
    // du/dx = 2*x*y, du/dxx = 2*y
    // du/dy = x^2 + 3*y^2, du/dyy = 6*y
    // du/dt = cos(t)
    auto coords = torch::tensor({{1.5f, 2.0f, 0.785398f},
                                 {0.5f, 1.0f, 0.0f}},
                                torch::requires_grad());

    auto x = coords.select(1, 0);
    auto y = coords.select(1, 1);
    auto t = coords.select(1, 2);

    auto u = x.pow(2) * y + y.pow(3) + torch::sin(t);

    auto du_vec = torch::autograd::grad({u}, {coords}, {torch::ones_like(u)}, true, true);
    auto du_dX = du_vec[0];

    auto du_dx = du_dX.select(1, 0);
    auto du_dy = du_dX.select(1, 1);
    auto du_dt = du_dX.select(1, 2);

    auto du_dxx_vec = torch::autograd::grad({du_dx}, {coords}, {torch::ones_like(du_dx)}, true, true);
    auto du_dxx = du_dxx_vec[0].select(1, 0);

    auto du_dyy_vec = torch::autograd::grad({du_dy}, {coords}, {torch::ones_like(du_dy)}, true, true);
    auto du_dyy = du_dyy_vec[0].select(1, 1);

    // Analytical checks for point 0: x=1.5, y=2.0, t=pi/4
    EXPECT_NEAR(du_dt[0].item<float>(), std::cos(0.785398f), 1e-4f);
    EXPECT_NEAR(du_dx[0].item<float>(), 2.0f * 1.5f * 2.0f, 1e-4f);
    EXPECT_NEAR(du_dy[0].item<float>(), (1.5f * 1.5f + 3.0f * 2.0f * 2.0f), 1e-4f);
    EXPECT_NEAR(du_dxx[0].item<float>(), 2.0f * 2.0f, 1e-4f);
    EXPECT_NEAR(du_dyy[0].item<float>(), 6.0f * 2.0f, 1e-4f);

    // Analytical checks for point 1: x=0.5, y=1.0, t=0.0
    EXPECT_NEAR(du_dt[1].item<float>(), 1.0f, 1e-4f);
    EXPECT_NEAR(du_dx[1].item<float>(), 2.0f * 0.5f * 1.0f, 1e-4f);
    EXPECT_NEAR(du_dxx[1].item<float>(), 2.0f * 1.0f, 1e-4f);
    EXPECT_NEAR(du_dyy[1].item<float>(), 6.0f * 1.0f, 1e-4f);
}

TEST(NetworkTest, PINNInitializationAndShape) {
    ThermalPINN model(3, 1, 64, 4);
    EXPECT_EQ(model->get_input_dim(), 3);
    EXPECT_EQ(model->get_output_dim(), 1);
    EXPECT_EQ(model->get_hidden_dim(), 64);
    EXPECT_GT(model->count_parameters(), 10000);

    auto x = torch::randn({32, 3});
    auto y = model->forward(x);
    EXPECT_EQ(y.sizes(), torch::IntArrayRef({32, 1}));
}

TEST(FDMTest, OpenMPGaussSeidelConvergenceAndAccuracy) {
    PDEDomain domain;
    domain.alpha = 0.05;
    domain.t_max = 0.1; // short test

    FDMConfig cfg;
    cfg.nx = 33;
    cfg.ny = 33;
    cfg.nt = 20;
    cfg.tolerance = 1e-8;

    OpenMPRedBlackGaussSeidelSolver solver(domain, cfg);
    auto res = solver.solve(/*verbose=*/false);

    EXPECT_EQ(res.x_coords.size(), 33);
    EXPECT_EQ(res.y_coords.size(), 33);
    EXPECT_EQ(res.u_final.size(), 33 * 33);
    EXPECT_GT(res.total_iterations, 0);

    // Check boundary points are 0
    for (int i = 0; i < cfg.nx; ++i) {
        EXPECT_NEAR(res.u_final[i * cfg.ny + 0], 0.0, 1e-12);
        EXPECT_NEAR(res.u_final[i * cfg.ny + (cfg.ny - 1)], 0.0, 1e-12);
    }
    for (int j = 0; j < cfg.ny; ++j) {
        EXPECT_NEAR(res.u_final[0 * cfg.ny + j], 0.0, 1e-12);
        EXPECT_NEAR(res.u_final[(cfg.nx - 1) * cfg.ny + j], 0.0, 1e-12);
    }

    // Check center point matches exact analytical decay to within 1%
    int mid_i = cfg.nx / 2;
    int mid_j = cfg.ny / 2;
    double exact_mid = domain.exact_solution(res.x_coords[mid_i], res.y_coords[mid_j], domain.t_max);
    double fdm_mid = res.u_final[mid_i * cfg.ny + mid_j];
    double rel_diff = std::abs(fdm_mid - exact_mid) / exact_mid;
    EXPECT_LT(rel_diff, 0.015);
}

TEST(ConservationTest, SimpsonIntegrationAndAnalyticalEnergy) {
    PDEDomain domain;
    int n = 65;
    std::vector<double> x(n), y(n), data(n * n);
    double h = 1.0 / (n - 1);
    for (int i = 0; i < n; ++i) {
        x[i] = i * h;
        y[i] = i * h;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            data[i * n + j] = std::sin(PI * x[i]) * std::sin(PI * y[j]);
        }
    }

    double computed_integral = ConservationValidator::integrate_2d(data, x, y);
    double exact_integral = 4.0 / (PI * PI);
    EXPECT_NEAR(computed_integral, exact_integral, 1e-6);
}

TEST(HDF5Test, SerializationSchemaVerification) {
    std::string test_path = "data/test_serialization.h5";
    ThermalFieldsPayload payload;
    payload.x_coords = {0.0, 0.5, 1.0};
    payload.y_coords = {0.0, 0.5, 1.0};
    payload.u_pinn.assign(9, 0.25);
    payload.u_fdm_gauss_seidel.assign(9, 0.24);
    payload.u_exact.assign(9, 0.245);
    payload.abs_error_pinn_vs_fdm.assign(9, 0.01);
    payload.pde_residual_map.assign(9, 0.001);
    payload.alpha = 0.05;
    payload.final_time = 0.5;
    payload.relative_l2_error = 0.012;
    payload.boundary_rmse = 0.001;
    payload.energy_conservation_rel_error = 0.004;

    bool ok = HDF5Serializer::serialize_fields(test_path, payload, /*verbose=*/false);
    EXPECT_TRUE(ok);
}

} // namespace pde_solver::testing
