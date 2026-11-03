import os
import sys
import json
import subprocess
import h5py
import numpy as np
import scipy.integrate
import scipy.fft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
H5_PATH = os.path.join(DATA_DIR, "thermal_fields_2d.h5")
ROOFLINE_JSON = os.path.join(ASSETS_DIR, "roofline_arithmetic_intensity.json")
BENCHMARK_JSON = os.path.join(ASSETS_DIR, "benchmark_summary.json")

def find_solver_binary():
    candidates = [
        os.path.join(PROJECT_ROOT, "build", "pde_solver_main.exe"),
        os.path.join(PROJECT_ROOT, "build", "pde_solver_main"),
        os.path.join(PROJECT_ROOT, "build", "bin", "pde_solver_main.exe"),
        os.path.join(PROJECT_ROOT, "build", "Release", "pde_solver_main.exe")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def run_solver_binary(binary_path):
    print(f">>> Invoking C++20 solver executable: {binary_path}")
    result = subprocess.run([binary_path], cwd=PROJECT_ROOT, capture_output=True, text=True)
    print(">>> C++ Solver Standard Output:")
    print(result.stdout)
    if result.returncode != 0:
        print(">>> C++ Solver Standard Error:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Solver failed with exit code {result.returncode}")

def evaluate_numerical_solution(h5_data):
    x = h5_data["x"]
    y = h5_data["y"]
    u_pinn = h5_data["u_pinn"]
    u_fdm = h5_data["u_fdm"]
    u_exact = h5_data["u_exact"]
    alpha = float(h5_data["attrs"]["alpha"])
    t_final = float(h5_data["attrs"]["final_time"])

    # 1. Relative L2 errors
    norm_exact = np.linalg.norm(u_exact)
    l2_pinn = np.linalg.norm(u_pinn - u_exact) / norm_exact
    l2_fdm = np.linalg.norm(u_fdm - u_exact) / norm_exact
    l2_diff = np.linalg.norm(u_pinn - u_fdm) / norm_exact

    # 2. Max pointwise error
    max_err_pinn = np.max(np.abs(u_pinn - u_exact))
    max_err_fdm = np.max(np.abs(u_fdm - u_exact))

    # 3. 2D Simpson integration via SciPy
    e_exact = (4.0 / (np.pi ** 2)) * np.exp(-2.0 * np.pi ** 2 * alpha * t_final)
    e_pinn = scipy.integrate.simpson(scipy.integrate.simpson(u_pinn, y), x)
    e_fdm = scipy.integrate.simpson(scipy.integrate.simpson(u_fdm, y), x)

    pinn_energy_err = abs(e_pinn - e_exact) / e_exact
    fdm_energy_err = abs(e_fdm - e_exact) / e_exact

    # 4. Type-I 2D DST solution of the Dirichlet heat equation
    X, Y = np.meshgrid(x, y, indexing="ij")
    u_initial = np.sin(np.pi * X) * np.sin(np.pi * Y)
    u0_hat = scipy.fft.dstn(u_initial[1:-1, 1:-1], type=1, norm="ortho")
    m, n = np.meshgrid(
        np.arange(1, u0_hat.shape[0] + 1),
        np.arange(1, u0_hat.shape[1] + 1),
        indexing="ij"
    )
    decay_kernel = np.exp(-alpha * (np.pi ** 2) * (m ** 2 + n ** 2) * t_final)
    u_spectral_int = scipy.fft.idstn(u0_hat * decay_kernel, type=1, norm="ortho")
    u_scipy_spectral = np.pad(u_spectral_int, ((1, 1), (1, 1)), mode="constant")
    spectral_diff = np.max(np.abs(u_scipy_spectral - u_exact))

    print("========================================================")
    print("       Python SciPy Field Evaluation and Spectral Check")
    print("========================================================")
    print(f"  Final Time T:               {t_final:.4f} s")
    print(f"  Thermal Diffusivity alpha:  {alpha:.4f}")
    print(f"  Relative L2 (PINN vs Exact): {l2_pinn:.4%}")
    print(f"  Relative L2 (FDM vs Exact):  {l2_fdm:.4%}")
    print(f"  Relative L2 (PINN vs FDM):   {l2_diff:.4%}")
    print(f"  Max Absolute Error (PINN):  {max_err_pinn:.4e}")
    print(f"  Max Absolute Error (FDM):   {max_err_fdm:.4e}")
    print(f"  Analytical Energy @ T:      {e_exact:.6f}")
    print(f"  SciPy Integrated PINN Energy: {e_pinn:.6f} (Rel Err: {pinn_energy_err:.4%})")
    print(f"  SciPy Integrated FDM Energy:  {e_fdm:.6f} (Rel Err: {fdm_energy_err:.4%})")
    print(f"  SciPy Spectral DST Max Diff:  {spectral_diff:.4e}")
    print("========================================================")

def generate_thermal_comparison_plot(h5_data, save_path):
    x = h5_data["x"]
    y = h5_data["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=200)

    # Global color scale for solutions
    vmin = 0.0
    vmax = float(np.max(h5_data["u_exact"]))

    # (a) Analytical Reference
    im0 = axes[0, 0].contourf(X, Y, h5_data["u_exact"], levels=50, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(r"(a) Analytical Exact Reference $u_{\rm exact}(x, y, T)$", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.85, label="Temperature $u$")

    # (b) OpenMP Red-Black Gauss-Seidel FDM
    im1 = axes[0, 1].contourf(X, Y, h5_data["u_fdm"], levels=50, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(r"(b) OpenMP Red-Black Gauss-Seidel FDM $u_{\rm FDM}(x, y, T)$", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.85, label="Temperature $u$")

    # (c) LibTorch PINN Surrogate
    im2 = axes[1, 0].contourf(X, Y, h5_data["u_pinn"], levels=50, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(r"(c) Native C++20 LibTorch PINN $u_{\rm PINN}(x, y, T)$", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.85, label="Temperature $u$")

    # (d) Absolute Error & PDE Residual
    abs_err = np.abs(h5_data["u_pinn"] - h5_data["u_exact"])
    im3 = axes[1, 1].contourf(X, Y, abs_err, levels=50, cmap="viridis")
    axes[1, 1].set_title(r"(d) Pointwise Absolute Error $|u_{\rm PINN} - u_{\rm exact}|$", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    fig.colorbar(im3, ax=axes[1, 1], shrink=0.85, label="Pointwise Error")

    plt.suptitle("LibTorch-PDE-Solver: 2D Transient Heat Diffusion Verification ($t = 0.5$, $\\alpha = 0.05$)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f">>> Generated plot: {save_path}")

def generate_roofline_plot(roofline_path, save_path):
    if not os.path.exists(roofline_path):
        print(f"Warning: Roofline JSON not found at {roofline_path}. Skipping plot.")
        return

    with open(roofline_path, "r") as f:
        data = json.load(f)

    fwd = data["forward_sweep"]
    res = data["residual_sweep"]

    batches = [pt["batch_size"] for pt in fwd]
    fwd_thru = [pt["throughput_pts_per_sec"] for pt in fwd]
    res_thru = [pt["throughput_pts_per_sec"] for pt in res]

    fwd_gflops = [pt["effective_gflops_per_sec"] for pt in fwd]
    res_gflops = [pt["effective_gflops_per_sec"] for pt in res]

    fwd_ai = [pt["arithmetic_intensity_flops_per_byte"] for pt in fwd]
    res_ai = [pt["arithmetic_intensity_flops_per_byte"] for pt in res]

    fwd_bw = [pt["effective_memory_bandwidth_gb_per_sec"] for pt in fwd]
    res_bw = [pt["effective_memory_bandwidth_gb_per_sec"] for pt in res]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    # Panel 1: Throughput & GFLOP/s vs Batch Size
    color = "tab:blue"
    ax1.set_xlabel("Collocation Batch Size $B$", fontsize=11, fontweight="bold")
    ax1.set_xscale("log", base=2)
    ax1.set_ylabel("Throughput (Collocation Points / sec)", color=color, fontsize=11, fontweight="bold")
    l1 = ax1.plot(batches, fwd_thru, "o-", color=color, linewidth=2, label="Forward Surrogate Throughput")
    l2 = ax1.plot(batches, res_thru, "s--", color="tab:cyan", linewidth=2, label="2nd-Order Autograd Throughput")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    color_gflops = "tab:red"
    ax1_twin.set_ylabel("Effective Compute (GFLOP/s)", color=color_gflops, fontsize=11, fontweight="bold")
    l3 = ax1_twin.plot(batches, fwd_gflops, "^-.", color=color_gflops, linewidth=2, label="Forward GFLOP/s")
    l4 = ax1_twin.plot(batches, res_gflops, "d:", color="tab:orange", linewidth=2, label="Autograd Residual GFLOP/s")
    ax1_twin.tick_params(axis="y", labelcolor=color_gflops)

    lines1 = l1 + l2 + l3 + l4
    labels1 = [l.get_label() for l in lines1]
    ax1.legend(lines1, labels1, loc="upper left", fontsize=9)
    ax1.set_title("Throughput & Compute Scaling across Batch Sizes", fontsize=12, fontweight="bold")

    # Panel 2: Memory Bandwidth & Arithmetic Intensity
    color_bw = "tab:purple"
    ax2.set_xlabel("Collocation Batch Size $B$", fontsize=11, fontweight="bold")
    ax2.set_xscale("log", base=2)
    ax2.set_ylabel("Memory Bandwidth (GB/s)", color=color_bw, fontsize=11, fontweight="bold")
    l5 = ax2.plot(batches, fwd_bw, "o-", color=color_bw, linewidth=2, label="Forward Memory Bandwidth")
    l6 = ax2.plot(batches, res_bw, "s--", color="mediumpurple", linewidth=2, label="Autograd Memory Bandwidth")
    ax2.tick_params(axis="y", labelcolor=color_bw)
    ax2.grid(True, alpha=0.3)

    ax2_twin = ax2.twinx()
    color_ai = "tab:green"
    ax2_twin.set_ylabel("Arithmetic Intensity (FLOPs / Byte)", color=color_ai, fontsize=11, fontweight="bold")
    l7 = ax2_twin.plot(batches, fwd_ai, "^-.", color=color_ai, linewidth=2, label="Forward AI")
    l8 = ax2_twin.plot(batches, res_ai, "d:", color="darkgreen", linewidth=2, label="Residual AI")
    ax2_twin.tick_params(axis="y", labelcolor=color_ai)

    ratios = [np.sqrt(batches[i + 1] / batches[i]) for i in range(len(batches) - 1)]
    edges = [batches[0] / ratios[0]]
    edges.extend(np.sqrt(batches[i] * batches[i + 1]) for i in range(len(batches) - 1))
    edges.append(batches[-1] * ratios[-1])
    combined_regimes = []
    for fwd_point, res_point in zip(fwd, res):
        fwd_regime = fwd_point["regime"]
        res_regime = res_point["regime"]
        combined_regimes.append(fwd_regime if fwd_regime == res_regime else "Mixed Regime")

    colors = {
        "Memory-Bandwidth Bound": ("gray", 0.15),
        "Compute-Bound": ("yellow", 0.10),
        "Mixed Regime": ("orange", 0.14)
    }
    regime_patches = []
    start = 0
    while start < len(combined_regimes):
        regime = combined_regimes[start]
        end = start + 1
        while end < len(combined_regimes) and combined_regimes[end] == regime:
            end += 1
        color, alpha = colors.get(regime, ("lightgray", 0.12))
        ax2.axvspan(edges[start], edges[end], color=color, alpha=alpha, zorder=0)
        if all(patch.get_label() != regime for patch in regime_patches):
            regime_patches.append(Patch(facecolor=color, alpha=alpha, label=regime))
        start = end

    lines2 = l5 + l6 + l7 + l8
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2 + regime_patches,
               labels2 + [patch.get_label() for patch in regime_patches],
               loc="lower right", fontsize=9)
    ax2.set_title("Memory Bandwidth & Arithmetic Intensity Profile", fontsize=12, fontweight="bold")

    plt.suptitle("Roofline Analysis & Memory-Bandwidth Scaling for LibTorch-PDE-Solver",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f">>> Generated plot: {save_path}")

def main():
    binary = find_solver_binary()
    if binary is not None and (not os.path.exists(H5_PATH) or "--run-solver" in sys.argv):
        run_solver_binary(binary)

    if not os.path.exists(H5_PATH):
        print(f"Error: Could not locate HDF5 file at {H5_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f">>> Loading HDF5 dataset: {H5_PATH}")
    with h5py.File(H5_PATH, "r") as f:
        h5_data = {
            "x": f["grid/x"][:],
            "y": f["grid/y"][:],
            "u_pinn": f["fields/u_pinn"][:],
            "u_fdm": f["fields/u_fdm_gauss_seidel"][:],
            "u_exact": f["fields/u_exact"][:],
            "abs_error": f["fields/abs_error_pinn_vs_fdm"][:],
            "pde_residual": f["fields/pde_residual_map"][:],
            "attrs": {k: f.attrs[k] for k in f.attrs.keys()}
        }

    evaluate_numerical_solution(h5_data)

    thermal_plot_path = os.path.join(ASSETS_DIR, "thermal_field_comparison.png")
    generate_thermal_comparison_plot(h5_data, thermal_plot_path)

    roofline_plot_path = os.path.join(ASSETS_DIR, "arithmetic_intensity_roofline.png")
    generate_roofline_plot(ROOFLINE_JSON, roofline_plot_path)

    print("Field evaluation and plotting finished.")

if __name__ == "__main__":
    main()
