#pragma once

#include <pybind11/pybind11.h>
#include "egret/coordinate.hpp"

namespace egret {

namespace py = pybind11;

// Abstract Element base class for pybind11 trampoline support.
// Methods return py::object to allow flexible Python overrides.
class Element {
public:
    virtual ~Element() = default;
    // Transfer a single coordinate: return a Python object (e.g., tuple)
    virtual py::object transfer(const Coordinate &cood0, double ds) = 0;
    // Return transfer_matrix_array-like result: (ndarray, s_list)
    virtual py::object transfer_array(const Coordinate &cood0, double ds, bool endpoint) = 0;
};

} // namespace egret
