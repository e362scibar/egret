/**
 * @file ring.hpp
 * @brief Ring element class definition
 * @author Hirokazu Maesaka
 * @date 2025
 */
// ring.hpp
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

#include "egret/element.hpp"

namespace egret {
    class Ring;
}
class egret::Ring: public egret::Element {
public:
    //! Factor for equilibrium emittance
    static constexpr double C_q = 3.83193864e-13;
    //! Electron rest mass in eV
    static constexpr double m_e_eV = 510998.95;
    //! Tolerance for finding initial coordinates of the closed orbit
    static double tol_cod;
    //! Maximum iterations for finding closed orbit
    static size_t max_iter_cod;

protected:
    //! Beam energy in eV
    double energy_;
    //! Betatron tune
    double tune_x_, tune_y_;
    //! Initial coordinates
    Coordinate cood0_;
    //! Initial envelope
    Envelope evlp0_;
    //! Initial dispersion
    Dispersion disp0_;
    //! Equilibrium emittance
    double emittance_x_, emittance_y_;
    //! Damping partition numbers
    double J_x_, J_y_, J_z_;

public:
    // Construct a new Ring object
    Ring(const std::string &name, const std::vector<std::shared_ptr<Element>> &elements,
        double energy, const std::string &info="") noexcept(false);
    /**
     * @brief Destroy the Ring object
     */
    virtual ~Ring() noexcept = default;

    /**
     * @brief Get the beam energy in eV.
     * @return double Beam energy
     */
    double energy() const { return energy_; }
    /**
     * @brief Get the horizontal betatron tune.
     * @return double Horizontal betatron tune
     */
    double tune_x() const { return tune_x_; }
    /**
     * @brief Get the vertical betatron tune.
     * @return double Vertical betatron tune
     */
    double tune_y() const { return tune_y_; }
    /**
     * @brief Get the initial coordinates.
     * @return const Coordinate& Initial coordinates
     */
    const Coordinate& cood0() const { return cood0_; }
    /**
     * @brief Get the initial envelope.
     * @return const Envelope& Initial envelope
     */
    const Envelope& evlp0() const { return evlp0_; }
    /**
     * @brief Get the initial dispersion.
     * @return const Dispersion& Initial dispersion
     */
    const Dispersion& disp0() const { return disp0_; }
    /**
     * @brief Get the equilibrium horizontal emittance.
     * @return double Equilibrium horizontal emittance
     */
    double emittance_x() const { return emittance_x_; }
    /**
     * @brief Get the equilibrium vertical emittance.
     * @return double Equilibrium vertical emittance
     */
    double emittance_y() const { return emittance_y_; }
    /**
     * @brief Get the horizontal damping partition number.
     * @return double Horizontal damping partition number
     */
    double J_x() const { return J_x_; }
    /**
     * @brief Get the vertical damping partition number.
     * @return double Vertical damping partition number
     */
    double J_y() const { return J_y_; }
    /**
     * @brief Get the longitudinal damping partition number.
     * @return double Longitudinal damping partition number
     */
    double J_z() const { return J_z_; }

    // Update the ring parameters (tunes, etc.)
    void update(double delta=0.0) noexcept(false);

    // find initial coordinates of closed orbit
    Coordinate find_initial_coordinate_of_closed_orbit(const Coordinate &cood_guess) const noexcept(false);

    // Polymorphic clone
    std::shared_ptr<Element> clone() const override;
};
