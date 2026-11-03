import os
import pytest
import h5py
import numpy as np
import scipy.integrate
import scipy.fft

HDF5_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "thermal_fields_2d.h5")

@pytest.fixture(scope="module")
def h5_data():
    if not os.path.exists(HDF5_PATH):
        pytest.skip(f"HDF5 file not found at {HDF5_PATH}. Run solver binary first.")
    with h5py.File(HDF5_PATH, "r") as f:
        data = {
            "x": f["grid/x"][:],
            "y": f["grid/y"][:],
            "u_pinn": f["fields/u_pinn"][:],
            "u_fdm": f["fields/u_fdm_gauss_seidel"][:],
            "u_exact": f["fields/u_exact"][:],
            "abs_error": f["fields/abs_error_pinn_vs_fdm"][:],
            "pde_residual": f["fields/pde_residual_map"][:],
            "attrs": {k: f.attrs[k] for k in f.attrs.keys()}
        }
    return data

def test_hdf5_schema_and_shapes(h5_data):
    assert h5_data["x"].shape == (129,)
    assert h5_data["y"].shape == (129,)
    assert h5_data["u_pinn"].shape == (129, 129)
    assert h5_data["u_fdm"].shape == (129, 129)
    assert h5_data["u_exact"].shape == (129, 129)
    assert h5_data["abs_error"].shape == (129, 129)
    assert h5_data["pde_residual"].shape == (129, 129)

def test_hdf5_attributes(h5_data):
    attrs = h5_data["attrs"]
    assert "alpha" in attrs
    assert "final_time" in attrs
    assert "relative_l2_error" in attrs
    assert "boundary_rmse" in attrs
    assert "energy_conservation_rel_error" in attrs
    assert np.isclose(attrs["alpha"], 0.05)
    assert np.isclose(attrs["final_time"], 0.5)

def test_pinn_relative_l2_accuracy(h5_data):
    u_pinn = h5_data["u_pinn"]
    u_exact = h5_data["u_exact"]
    rel_l2 = np.linalg.norm(u_pinn - u_exact) / np.linalg.norm(u_exact)
    assert rel_l2 < 0.015, f"Relative L2 error ({rel_l2:.4%}) exceeds 1.5% threshold"

def test_fdm_relative_l2_accuracy(h5_data):
    u_fdm = h5_data["u_fdm"]
    u_exact = h5_data["u_exact"]
    rel_l2 = np.linalg.norm(u_fdm - u_exact) / np.linalg.norm(u_exact)
    assert rel_l2 < 0.010, f"FDM Relative L2 error ({rel_l2:.4%}) exceeds 1.0% threshold"

def test_boundary_condition_fidelity(h5_data):
    u_pinn = h5_data["u_pinn"]
    bc_vals = np.concatenate([
        u_pinn[0, :],
        u_pinn[-1, :],
        u_pinn[:, 0],
        u_pinn[:, -1]
    ])
    max_bc_err = np.max(np.abs(bc_vals))
    rmse_bc = np.sqrt(np.mean(bc_vals ** 2))
    assert max_bc_err < 0.05, f"Boundary max error ({max_bc_err:.4e}) exceeds 0.05"
    assert rmse_bc < 0.015, f"Boundary RMSE ({rmse_bc:.4e}) exceeds 0.015"

def test_scipy_simpson_energy_conservation(h5_data):
    x = h5_data["x"]
    y = h5_data["y"]
    u_pinn = h5_data["u_pinn"]
    u_fdm = h5_data["u_fdm"]
    alpha = float(h5_data["attrs"]["alpha"])
    t_final = float(h5_data["attrs"]["final_time"])

    # 2D Simpson integration via SciPy
    e_pinn = scipy.integrate.simpson(scipy.integrate.simpson(u_pinn, y), x)
    e_fdm = scipy.integrate.simpson(scipy.integrate.simpson(u_fdm, y), x)
    e_exact = (4.0 / (np.pi ** 2)) * np.exp(-2.0 * np.pi ** 2 * alpha * t_final)

    rel_err_pinn = abs(e_pinn - e_exact) / e_exact
    rel_err_fdm = abs(e_fdm - e_exact) / e_exact

    assert rel_err_pinn < 0.01, f"PINN energy relative error ({rel_err_pinn:.4%}) exceeds 1%"
    assert rel_err_fdm < 0.01, f"FDM energy relative error ({rel_err_fdm:.4%}) exceeds 1%"

def test_scipy_spectral_dst_cross_check(h5_data):
    x = h5_data["x"]
    y = h5_data["y"]
    alpha = float(h5_data["attrs"]["alpha"])
    t_final = float(h5_data["attrs"]["final_time"])
    nx, ny = len(x), len(y)

    # Solve each Dirichlet sine mode through its spectral decay rate.
    X, Y = np.meshgrid(x, y, indexing="ij")
    u_initial = np.sin(np.pi * X) * np.sin(np.pi * Y)
    u0_hat = scipy.fft.dstn(u_initial[1:-1, 1:-1], type=1, norm="ortho")
    m, n = np.meshgrid(
        np.arange(1, nx - 1),
        np.arange(1, ny - 1),
        indexing="ij"
    )
    decay_kernel = np.exp(-alpha * (np.pi ** 2) * (m ** 2 + n ** 2) * t_final)
    u_spectral_int = scipy.fft.idstn(u0_hat * decay_kernel, type=1, norm="ortho")
    u_spectral = np.pad(u_spectral_int, ((1, 1), (1, 1)), mode="constant")

    diff = np.max(np.abs(u_spectral - h5_data["u_exact"]))
    assert diff < 1e-14, f"Spectral DST reference differs from exact analytical solution by {diff}"
