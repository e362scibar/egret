cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
option(USE_OPENMP "Enable OpenMP parallelization in pyegret helper" OFF)

# Prefer system-installed pybind11; fall back to requiring it to be available
option(BUILD_PYEXT "Build Python extension with pybind11" OFF)
if(BUILD_PYEXT)
  find_package(pybind11 CONFIG REQUIRED)
endif()

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(egret_core STATIC
  src/coordinate.cpp
  src/coordinatearray.cpp
  src/drift.cpp
  src/steering.cpp
  src/quadrupole.cpp
  src/dipole.cpp
  src/sextupole.cpp
)
target_include_directories(egret_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(egret_core PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)
target_link_libraries(egret_core PUBLIC Eigen3::Eigen)

if(BUILD_PYEXT)
  pybind11_add_module(pyegret src/bindings.cpp)
  target_link_libraries(pyegret PRIVATE egret_core)
  target_include_directories(pyegret PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_include_directories(pyegret PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)
  if(USE_OPENMP)
    target_link_libraries(pyegret PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(pyegret PRIVATE USE_OPENMP)
  endif()
  # Install python extension into a relative `egret` directory under the
  # install prefix. scikit-build-core sets the install prefix to a temporary
  # directory when building a wheel, so installing to the relative path
  # `egret/` ensures the shared object ends up in the package directory.
  install(TARGETS pyegret
    LIBRARY DESTINATION egret
    RUNTIME DESTINATION egret
    ARCHIVE DESTINATION egret)
endif()

add_executable(egret_cli src/cli_quadrupole.cpp)
target_link_libraries(egret_cli PRIVATE egret_core Eigen3::Eigen)
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
