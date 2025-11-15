// element.hpp
//
// Copyright (C) 2025 Hirokazu Maesaka (RIKEN SPring-8 Center)
//
// This file is part of Egret: Engine for General Research in
// Energetic-beam Tracking.
//
// Egret is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
