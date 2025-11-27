/**
 * @file lattice.cpp
 * @brief Lattice element class implementation
 * @author Hirokazu Maesaka
 * @date 2025
 */
// lattice.cpp
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

#include "egret/lattice.hpp"
#include <ranges>

/**
 * @brief Construct a new egret::Lattice object.
 * @param name Lattice name
 * @param elements Vector of child elements
 * @param dx Horizontal offset
 * @param dy Vertical offset
 * @param ds Longitudinal offset
 * @param tilt Tilt angle
 * @param info Additional info string
 */
egret::Lattice::Lattice(const std::string &name,
    const std::vector<std::shared_ptr<Element>> &elements,
    const double dx, const double dy, const double ds,
    const double tilt, const std::string &info) noexcept(false) :
    Element(name, Lattice::length(elements), Lattice::angle(elements),
        dx, dy, ds, tilt, info) {
    // Deep-copy incoming elements by calling their polymorphic clone()
    elements_ = std::vector<std::shared_ptr<Element>>();
    for (const auto &elem: elements) {
        elements_->push_back(elem->clone());
    }
}

// Polymorphic clone implementation for Lattice: clone children as well.
std::shared_ptr<egret::Element> egret::Lattice::clone() const {
    /*
    std::vector<std::shared_ptr<Element>> elems;
    if (elements_) {
        for (const auto &e : *elements_) {
            elems.push_back(e->clone());
        }
    }
    */
    auto newlat = std::make_shared<Lattice>(name(), elements_, dx(), dy(), ds(), tilt(), info());
    // copy indices if needed
    //newlat->set_indices();
    return newlat;
}

/**
 * @brief Calculate the total length of a vector of elements.
 * @param elements Vector of elements
 * @return double Total length
 */
double egret::Lattice::length(const std::vector<std::shared_ptr<Element>> &elements) noexcept {
    double length = 0.0;
    for (const auto &elem: elements) {
        length += elem->length();
    }
    return length;
}

/**
 * @brief Calculate the total angle of a vector of elements.
 * @param elements Vector of elements
 * @return double Total angle
 */
double egret::Lattice::angle(const std::vector<std::shared_ptr<Element>> &elements) noexcept {
    double angle = 0.0;
    for (const auto &elem: elements) {
        angle += elem->angle();
    }
    return angle;
}


