# LibTorch PDE Solver

A native C++20 Scientific ML & numerical PDE solver benchmarking a LibTorch Physics-Informed Neural Network (`torch::autograd::grad` higher-order spatio-temporal residuals for the 2D heat diffusion equation) against a parallel OpenMP Red-Black Gauss-Seidel Finite-Difference (FDM) baseline with HDF5 serialization. The project also writes HDF5 field data and verifies the result with Python, SciPy, and h5py.

## Problem

The solver uses thermal diffusivity $\alpha = 0.05$ on $x,y \in [0,1]$ and $t \in [0,0.5]$:

$$
\frac{\partial u}{\partial t} - \alpha\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)=0.
$$

The initial condition is $u(x,y,0) = \sin(\pi x)\sin(\pi y)$. All four spatial boundaries use homogeneous Dirichlet conditions. The analytical solution is

$$
u(x,y,t)=e^{-2\alpha\pi^2t}\sin(\pi x)\sin(\pi y).
$$

For this boundary value problem, thermal energy decays according to the analytical Dirichlet dissipation law $E(t) = E(0)e^{-2\alpha\pi^2t}$. The validator compares numerical energy against that decay law; it does not treat energy as constant.

## Implementation

- **PINN:** A `3 → 64 → 64 → 64 → 64 → 1` Tanh network. `torch::autograd::grad` computes first and second derivatives for the PDE residual. Training uses fixed-seed stratified collocation, initial-condition, and boundary samples, and restores the checkpoint with the lowest relative L2 error on a 129 × 129 validation grid.
- **FDM:** Backward Euler in time with central differences in space. Red and black cells are updated in separate OpenMP sweeps. The stopping residual is the maximum update over both colors.
- **Validation:** Relative L2 and maximum absolute errors against the analytical field, PINN boundary fidelity, and Simpson-integrated energy-decay agreement.
- **Profiling:** The profiler times forward and second-order residual passes over five batch sizes. FLOP counts and bytes are analytical estimates; GFLOP/s and effective GB/s combine those estimates with measured latency. They are not hardware-counter measurements.
- **HDF5:** C++ writes coordinates and five 129 × 129 fields to `data/thermal_fields_2d.h5`; Python reads and verifies the file.
- **Spectral check:** The Python verifier performs a Type-I 2D discrete sine transform on the 127 × 127 interior field, applies the modal heat-equation propagator, then applies the inverse transform and restores zero boundaries.

## Layout

```text
LibTorch-PDE-Solver/
├── CMakeLists.txt
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/solver_config.yaml
├── include/pde_solver/
├── src/
│   └── main_solver.cpp
├── tests/
│   ├── cpp/test_pde_components.cpp
│   └── python/test_hdf5_and_spectral.py
├── scripts/evaluate_and_plot.py
├── data/thermal_fields_2d.h5
└── assets/
    ├── benchmark_summary.json
    ├── roofline_arithmetic_intensity.json
    ├── thermal_field_comparison.png
    └── arithmetic_intensity_roofline.png
```

## Recorded results

The checked-in summary was generated with seed 42, 2,400 PINN epochs, one LibTorch CPU thread, and eight OpenMP FDM threads. Latency and throughput depend on the host and system load; the seeded numerical fields and validation metrics are repeatable with the same software and execution environment.

### Accuracy and runtime

| Metric | FDM | PINN | Notes |
|---|---:|---:|---:|
| Grid | 129 × 129 | 129 × 129 | 16,641 points |
| Time steps / horizon | 200 | Continuous time | $t \in [0, 0.5]$ |
| Training epochs / best epoch | — | 2,400 / 2,400 | Seed 42 |
| Training time | — | 89.095872 s | Measured on this host |
| Solve time | 3.376372 s | — | 200 FDM steps |
| Mean Gauss-Seidel iterations per step | 57.83 | — | Tolerance $10^{-9}$ |
| Relative L2 error vs. exact | 0.0634% | 1.2574% | Measured on the 129 × 129 grid |
| Relative L2 error vs. FDM | — | 1.2720% | — |
| Maximum absolute error vs. exact | 0.000387 | 0.029869 | — |
| PINN boundary maximum / RMSE | 0 | 0.029869 / 0.005865 | Homogeneous Dirichlet boundary |
| Analytical energy at $t = 0.5$ | 0.247426 | 0.247426 | Analytical decay law |
| Integrated energy at $t = 0.5$ | 0.247582 | 0.247747 | FDM / PINN |
| Relative energy-decay error | 0.0634% | 0.1301% | Compared with analytical energy decay |
| Solve throughput | 985,732.556381 points/s | 533,990.084554 points/s | FDM space-time points; PINN final-grid points |
| Final-grid evaluation time | — | 31.1635 ms | Single PINN pass |
| FDM solve time / PINN final-grid pass | — | 108.343806× | Excludes PINN training; workloads differ |

Energy values report agreement with the analytical decay law. The FDM throughput counts space-time points across all time steps; the PINN throughput counts one final-grid evaluation. The ratio above is a timing comparison, not a like-for-like end-to-end speedup.

### Roofline batch sweep

The model has 12,801 parameters occupying 51,204 bytes (**50.0 KiB**). The ridge point is 13 FLOP/byte. The reported arithmetic intensity uses estimated activation, derivative, weight, and gradient-sized traffic. The effective bandwidth and GFLOP/s values use that estimated traffic and the measured mean latency.

| Pass | Batch | Mean latency (ms) | Throughput (points/s) | FLOP/byte | GFLOP/s | Effective GB/s | Regime |
|---|---:|---:|---:|---:|---:|---:|---|
| Forward | 256 | 1.45124 | 176,401 | 12.0993 | 4.83216 | 0.399375 | Memory-Bandwidth Bound |
| Forward | 1,024 | 4.96358 | 206,303 | 12.9579 | 5.65125 | 0.436125 | Memory-Bandwidth Bound |
| Forward | 4,096 | 20.6118 | 198,721 | 13.1919 | 5.44357 | 0.412645 | Compute-Bound |
| Forward | 16,384 | 84.4483 | 194,012 | 13.2517 | 5.31457 | 0.401047 | Compute-Bound |
| Forward | 65,536 | 296.116 | 221,319 | 13.2668 | 6.06259 | 0.456975 | Compute-Bound |
| Residual | 256 | 4.90195 | 52,224.1 | 12.6224 | 7.86816 | 0.623348 | Memory-Bandwidth Bound |
| Residual | 1,024 | 17.9324 | 57,103.5 | 12.9479 | 8.60329 | 0.664456 | Memory-Bandwidth Bound |
| Residual | 4,096 | 75.0449 | 54,580.7 | 13.0319 | 8.22321 | 0.631007 | Compute-Bound |
| Residual | 16,384 | 195.097 | 83,978.7 | 13.053 | 12.6524 | 0.969303 | Compute-Bound |
| Residual | 65,536 | 1,000.06 | 65,531.8 | 13.0583 | 9.87313 | 0.756078 | Compute-Bound |

The visualization shades each batch range using the regime labels in `assets/roofline_arithmetic_intensity.json`.

## Build and run

Run these commands from the project root. The C++ build requires CMake 3.20+, C++20, OpenMP, HDF5, GoogleTest, and LibTorch built for the selected compiler and platform. Python verification requires Python 3.10+.

### Linux — Ubuntu / Debian

Install the system dependencies and obtain the CPU LibTorch C++ distribution from [PyTorch](https://pytorch.org/get-started/locally/). Set `LIBTORCH` to the extracted directory.

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libomp-dev libhdf5-dev libgtest-dev python3-venv python3-pip

export LIBTORCH=/path/to/libtorch
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LIBTORCH"
cmake --build build --parallel
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"

ctest --test-dir build --output-on-failure
./build/pde_solver_main --config configs/solver_config.yaml

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/evaluate_and_plot.py
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider
```

Set `Torch_DIR` to the directory containing `TorchConfig.cmake` instead of `CMAKE_PREFIX_PATH` if LibTorch is installed separately from the rest of the prefix.

### Windows — MSYS2 UCRT64

Open the **MSYS2 UCRT64** shell. Install the compiler, libraries, LibTorch package, and Python dependencies with pacman:

```bash
pacman -S --needed \
  mingw-w64-ucrt-x86_64-toolchain \
  mingw-w64-ucrt-x86_64-cmake \
  mingw-w64-ucrt-x86_64-ninja \
  mingw-w64-ucrt-x86_64-hdf5 \
  mingw-w64-ucrt-x86_64-gtest \
  mingw-w64-ucrt-x86_64-python \
  mingw-w64-ucrt-x86_64-python-pytorch \
  mingw-w64-ucrt-x86_64-python-numpy \
  mingw-w64-ucrt-x86_64-python-scipy \
  mingw-w64-ucrt-x86_64-python-h5py \
  mingw-w64-ucrt-x86_64-python-matplotlib \
  mingw-w64-ucrt-x86_64-python-pytest

TORCH_LIB="$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")"
export PATH="$TORCH_LIB:/ucrt64/bin:$PATH"

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

ctest --test-dir build --output-on-failure
./build/pde_solver_main.exe --config configs/solver_config.yaml

python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/evaluate_and_plot.py
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider
```

With the MSYS2 UCRT64 compiler, CMake locates the matching LibTorch package automatically when `Torch_DIR` is not set. Other installations can provide `Torch_DIR` or `CMAKE_PREFIX_PATH` explicitly.

The solver accepts `--config <path>`. Without it, it loads `configs/solver_config.yaml` from the project root or `../configs/solver_config.yaml` when launched from `build/`. The configuration file controls the domain, grid, training parameters, seed, profiler batches, and output paths.

## License

This project is distributed under the [MIT License](LICENSE).

## References & Acknowledgments

The PINN formulation follows M. Raissi, P. Perdikaris, and G. E. Karniadakis, [“Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations”](https://doi.org/10.1016/j.jcp.2018.10.045), *Journal of Computational Physics* 378 (2019), 686–707. Python-side field checks use [SciPy](https://github.com/scipy/scipy) for Simpson integration and discrete sine transforms.
