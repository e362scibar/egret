# egret C++ core (pyegret)

This folder contains the C++ core of the `egret` numerical kernels and a pybind11 extension `pyegret` used by the Python bridge.

Quick build & test (local)

Prerequisites:

- CMake >= 3.15
- A C++ compiler supporting C++20
- Eigen3 development headers
- Python 3.10, `pybind11`, `numpy`, and `pytest` installed in your Python environment

Build and run tests (example using system Python):

```bash
cd egret/cpp
PYBIND11_DIR=$(python -m pybind11 --cmakedir)
cmake -S . -B build -DBUILD_PYEXT=ON -DUSE_OPENMP=ON -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build build -j 4

# Run pytest (ensure PYTHONPATH includes repo root and build dir)
export PYTHONPATH=$(pwd)/..:$(pwd)/build:$PYTHONPATH
python -m pytest -q egret/cpp/tests
```

Benchmarks

- A small benchmark script is provided at `egret/cpp/benchmarks/benchmark_apply.py`.
- Run it with `PYTHONPATH` set as above.

Notes

- `USE_OPENMP` is a CMake option that enables OpenMP parallel loops; on macOS OpenMP support is more complex — CI disables OpenMP on macOS by default. Adjust `-DUSE_OPENMP=ON|OFF` when configuring.
- Runtime heuristics in the binding currently control whether threads are used based on work size; thresholds are in `bindings.cpp` and can be made configurable if desired.

If you want me to add a small `pyegret` packaging step (wheel) or make the runtime thresholds configurable via environment variables, tell me and I will add it.

## libegret (C++ core) — README

What this scaffold provides

- A minimal C++ core (`egret_core`) implementing `Coordinate` and `CoordinateArray`.
- Simple implementations of `Drift` transfer matrices and transfer-matrix arrays.
- A `pyegret` Python extension built with `pybind11` exposing basic types.

Targets & toolchains

- Primary target: Linux x86_64 and aarch64 (Linux on ARM). You requested Apple Silicon (aarch64) support — see notes below.
- C++ standard: C++20. Python: 3.10+.

### Building locally (Linux x86_64)

Prerequisites:

- CMake >= 3.20
- A C++20 compiler (g++ or clang)
- Eigen3 development headers

Example build:

```bash
mkdir -p cpp/build && cd cpp/build
cmake ..
cmake --build . --config Release -j
```

This will build `libegret` and the Python extension `pyegret` (if a Python dev environment is available).

### Cross-building for Linux aarch64 (CI / cross toolchain)

Option 1 — cross-build with a toolchain:

- Install an aarch64 cross toolchain (`aarch64-linux-gnu-gcc` etc.) and provide CMake a toolchain file.
- Example (toolchain file `aarch64-toolchain.cmake`):

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
```

Then run:

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=../aarch64-toolchain.cmake ..
cmake --build .
```

Option 2 — build on an ARM runner / machine:

- Build natively on an ARM machine (recommended for correctness and SIMD testing).

### Apple Silicon (macOS on aarch64 / arm64)

- Cross-building Linux aarch64 is different from building macOS aarch64 (Apple Silicon). If your target is macOS on Apple Silicon, prefer building natively on an Apple Silicon machine or using CI macOS runners.
- macOS cross-compilation from Linux is more complex (different sysroot/SDK).

CI recommendations

- For Linux x86_64 and aarch64: Use GitHub Actions with `ubuntu-latest` for x86_64 and `ubuntu-22.04` self-hosted aarch64 runner or cross-toolchain matrix.
- For Apple Silicon macOS wheels: use `macos-13` (or macos-12) runner on GitHub Actions (check whether runner is arm64 — GitHub provides arm64 macOS runners or you can use self-hosted Apple Silicon machines).

Next steps

- Implement more numeric kernels (transfer/transfer_array for quadrupoles, dipoles, sextupoles).
- Add unit tests that compare Python reference outputs with C++ results.
- Improve data interchange (zero-copy Eigen <-> numpy) for performance.
