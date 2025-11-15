/**
 * @brief Base class for all objects in the Egret framework.
 * Provides a common interface for naming and identification.
 */
// object.hpp
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

#include <string>

namespace egret {

/**
 * @brief Base class for all objects in the Egret framework.
 * Provides a common interface for naming and identification.
 */
class Object {
protected:
    //! Name of the object.
    std::string name;
public:
    /**
     * @brief Construct a new Object object
     *
     * @param name Object name
     */
    Object(const std::string &name) : name(name) {}
    /**
     * @brief Virtual destructor
     */
    virtual ~Object() = default;
    /**
     * @brief Get the name of the object
     *
     * @return const std::string& Name of the object
     */
    const std::string& get_name() const { return name; }
    /**
     * @brief Set the name of the object
     *
     * @param new_name New name to set
     */
    void set_name(const std::string &new_name) { name = new_name; }
};

} // namespace egret
