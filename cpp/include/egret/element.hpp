/**
 * @brief Base class for an accelerator element in the Egret framework.
 */
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
#include "egret/object.hpp"
#include "egret/coordinate.hpp"
#include "egret/coordinatearray.hpp"
#include "egret/envelope.hpp"
#include "egret/envelopearray.hpp"
#include "egret/dispersion.hpp"
#include "egret/dispersionarray.hpp"

namespace egret {

namespace py = pybind11;

/**
 * @brief Base class for an accelerator element in the Egret framework.
 */
class Element: public Object {
protected:
    //! Length of the element
    double length;
    //! Transverse and longitudinal offsets and tilt
    double dx, dy, ds, tilt;
    //! Additional info string
    std::string info;
public:
    /**
     * @brief Construct a new Element object
     *
     * @param name Object name
     * @param length Length of the element
     * @param dx Transverse offset in x
     * @param dy Transverse offset in y
     * @param ds Longitudinal offset
     * @param tilt Tilt angle
     * @param info Additional info string
     */
    Element(const std::string &name, double length,
            double dx=0.0, double dy=0.0, double ds=0.0, double tilt=0.0,
            const std::string &info="")
        : Object(name), length(length), dx(dx), dy(dy), ds(ds), tilt(tilt), info(info) {}
    /**
     * @brief Virtual destructor
     */
    virtual ~Element() = default;
    /**
     * @brief Transfer a single coordinate through the element
     *
     * @param cood0 Initial coordinate
     * @param ds Maximum step size
     * @return Coordinate Transferred coordinate
     */
    virtual std::Tuple<Coordinate, Envelope, Dispersion>
        transfer(const Coordinate &cood0, const Envelope &evlp0, Dispersion &disp0, const double ds);
    virtual std::Tuple<CoordinateArray, EnvelopeArray, DispersionArray>
        transfer_array(const Coordinate &cood0, const Envelope &evlp0, const Dispersion &disp0, const double ds, bool endpoint);

    // Return transfer matrix
    virtual py::object transfer(const Coordinate &cood0, double ds) = 0;
    // Return transfer_matrix_array-like result: (ndarray, s_list)
    virtual py::object transfer_array(const Coordinate &cood0, double ds, bool endpoint) = 0;
};

} // namespace egret
