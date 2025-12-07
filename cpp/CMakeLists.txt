cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Prefer g++-12 or newer if available, otherwise fall back to system default
if(NOT CMAKE_CXX_COMPILER)
  find_program(GXX_12_OR_NEWER NAMES g++-14 g++-13 g++-12)
  if(GXX_12_OR_NEWER)
    set(CMAKE_CXX_COMPILER ${GXX_12_OR_NEWER})
    message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER}")
  endif()
endif()

# Configure common warning flags but do NOT force -Werror globally. For
# packaging builds (PEP 517) global -Werror can cause LTO / compiler-time
# warnings (e.g. Eigen false-positives) to turn into fatal link errors.
set(DEFAULT_WARNING_FLAGS "-Wall -Wextra -Wpedantic")
string(APPEND CMAKE_CXX_FLAGS " ${DEFAULT_WARNING_FLAGS}")

# Optionally enable treating warnings as errors. Use this during local dev
# builds by passing -DWARNINGS_AS_ERRORS=ON to cmake if desired.
option(WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
if(WARNINGS_AS_ERRORS)
  string(APPEND CMAKE_CXX_FLAGS " -Werror")
endif()

# Try to extract Python package version from pyproject.toml so the C++ code
# and package version remain in sync. If parsing fails, fall back to 0.0.0.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../pyproject.toml" _egret_pyproj CONTENTS)
string(REGEX MATCH "version[ \t]*=[ \t]*\"[^\"]+\"" _verline "${_egret_pyproj}")
if(_verline)
    string(REGEX REPLACE "^.*version[ \t]*=[ \t]*\"([^\"]+)\".*$" "\\1" EGRET_PY_VERSION "${_verline}")
else()
    set(EGRET_PY_VERSION "0.0.0")
endif()

project(libegret VERSION ${EGRET_PY_VERSION} LANGUAGES CXX)

# Generate a small header exposing the Python package version to C++.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/egret/version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/egret/version.hpp"
  @ONLY
)

# Option to enable OpenMP parallelization in the Python extension
option(USE_OPENMP "Enable OpenMP parallelization in cppegret helper" OFF)

# Prefer system-installed pybind11; fall back to requiring it to be available
option(BUILD_PYEXT "Build Python extension with pybind11" OFF)
if(BUILD_PYEXT)
  # Prefer using CMake's modern FindPython to avoid pybind11 triggering
  # legacy FindPythonInterp/FindPythonLibs behavior. Set policy CMP0148
  # to NEW so CMake does not emit the dev warning about removed modules.
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
  endif()

  # Resolve Python explicitly (Interpreter + Development) prior to invoking
  # pybind11. This prevents pybind11 from falling back to older FindPython
  # modules and eliminates the CMP0148 warning.
  find_package(Python COMPONENTS Interpreter Development REQUIRED)

  # Instruct pybind11's CMake helpers to use the modern FindPython results
  # instead of trying legacy FindPythonInterp/FindPythonLibs.
  set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)

  find_package(pybind11 CONFIG REQUIRED)
endif()

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(GSL REQUIRED)

add_library(egret_core STATIC
  src/basearray.cpp
  src/envelope.cpp
  src/coordinatearray.cpp
  src/dispersionarray.cpp
  src/envelopearray.cpp
  src/element.cpp
  src/drift.cpp
  src/steering.cpp
  src/dipole.cpp
  src/quadrupole.cpp
  src/nonlinearmultipole.cpp
  src/sextupole.cpp
  src/octupole.cpp
  src/lattice.cpp
  src/ring.cpp
)
target_include_directories(egret_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(egret_core PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)
target_link_libraries(egret_core PUBLIC Eigen3::Eigen)
# Initialize Eigen matrices by zero to avoid spurious -Wmaybe-uninitialized warnings
target_compile_definitions(egret_core PUBLIC EIGEN_INITIALIZE_MATRICES_BY_ZERO=1)
# Ensure egret_core is position-independent so it can be linked into the
# Python extension (shared object) without PIC-related link errors.
set_target_properties(egret_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Link GSL to the core library so consumers (eg. the pybind11 module)
# do not need to manually link GSL and to avoid unresolved symbols at link.
target_link_libraries(egret_core PUBLIC GSL::gsl GSL::gslcblas)

if(BUILD_PYEXT)
  pybind11_add_module(cppegret src/bindings.cpp)
  target_link_libraries(cppegret PRIVATE egret_core)
  target_include_directories(cppegret PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_include_directories(cppegret PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)
  # Ensure the pybind11 extension also uses zero-initialized Eigen matrices
  target_compile_definitions(cppegret PRIVATE EIGEN_INITIALIZE_MATRICES_BY_ZERO=1)
  if(USE_OPENMP)
    target_link_libraries(cppegret PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(cppegret PRIVATE USE_OPENMP)
  endif()
  # Install python extension into a relative `egret` directory under the
  # install prefix. scikit-build-core sets the install prefix to a temporary
  # directory when building a wheel, so installing to the relative path
  # `egret/` ensures the shared object ends up in the package directory.
  install(TARGETS cppegret
    LIBRARY DESTINATION egret
    RUNTIME DESTINATION egret
    ARCHIVE DESTINATION egret)
endif()

add_executable(egret_cli src/cli_quadrupole.cpp)
target_link_libraries(egret_cli PRIVATE egret_core Eigen3::Eigen GSL::gsl GSL::gslcblas)
target_include_directories(egret_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(egret_cli PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)

install(TARGETS egret_core EXPORT egretTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/libegretConfigVersion.cmake"
  VERSION "${EGRET_PY_VERSION}"
  COMPATIBILITY AnyNewerVersion
)
