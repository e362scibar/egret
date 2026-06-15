# Egret

- Egret is a general-purpose beam dynamics simulation code.
- Egret stands for Energetic General Research in Energetic-beam Tracking.
- Author: Hirokazu Maesaka (RIKEN SPring-8 Center)

## Dependencies

### C++ (build-time)

| Library | Version | Notes |
|---------|---------|-------|
| Eigen3 | ≥ 3.3 | Linear algebra throughout |
| GSL (GNU Scientific Library) | any | ODE integration and numerical routines |
| OpenMP | — | Optional; enables parallel tracking (`-DUSE_OPENMP=ON`) |
| CMake | ≥ 3.20 | Build system |
| C++20-compatible compiler | — | e.g. GCC 12+, Clang 14+ |

On Ubuntu/Debian:
```bash
sudo apt install libeigen3-dev libgsl-dev cmake ninja-build
```
On macOS (Homebrew):
```bash
brew install eigen gsl cmake ninja
```

### Python (runtime)

| Package | Notes |
|---------|-------|
| numpy | Array I/O and numerical interface |
| latticejson | LatticeJSON ring file parsing (`Ring.read_json()`) |

### Python (build-time only)

| Package | Notes |
|---------|-------|
| pybind11 ≥ 3.0 | C++/Python binding generator |
| scikit-build-core ≥ 0.11.6 | PEP 517 build backend (drives CMake) |
| scipy | Used during the build process |
| wheel, setuptools | Standard packaging tools |

## Building (developer notes)

This package uses CMake + pybind11 for the C++ core and `scikit-build-core` as
the PEP 517 build backend. The C++ sources live under the `cpp/` directory and
are built automatically when creating a wheel.

Recommended local build (creates a wheel in `../dist_wheels`):

```bash
cd /path/to/repo/egret
python -m build -w -o ../dist_wheels
```

Quick smoke test (install the wheel into a fresh venv and verify import):

```bash
python -m venv .venv
source .venv/bin/activate
pip install ../dist_wheels/egret-0.1.6-*.whl
python -c "import egret; import importlib; m=importlib.import_module('egret._pyegret_bridge'); print('PYEGRET', getattr(m,'_PYEGRET_AVAILABLE',None))"
```

Notes:

- `setup.py` has been removed; use the PEP 517 flow above.
- `scikit-build-core` will inject `cmake` and `ninja` into isolated build environments automatically. You do not need to
 include `cmake`/`ninja` in `build-system.requires` of `pyproject.toml`; the project already relies on `scikit-build-core`.
- If you need to run the CMake build manually (development), you can still run
 a CMake build under `egret/cpp` and copy the produced `pyegret` shared object
 into the `egret/` package directory for local testing.
