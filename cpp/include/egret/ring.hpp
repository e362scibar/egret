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
    double Jx_, Jy_, Jz_;
    //! Radiation integrals
    double I2_, I4_, I5u_, I5v_, I4u_, I4v_;

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
    double Jx() const { return Jx_; }
    /**
     * @brief Get the vertical damping partition number.
     * @return double Vertical damping partition number
     */
    double Jy() const { return Jy_; }
    /**
     * @brief Get the longitudinal damping partition number.
     * @return double Longitudinal damping partition number
     */
    double Jz() const { return Jz_; }
    /**
     * @brief Get the second radiation integral I2.
     * @return double Radiation integral I2
     */
    double I2() const { return I2_; }
    /**
     * @brief Get the fourth radiation integral I4.
     * @return double Radiation integral I4
     */
    double I4() const { return I4_; }
    /**
     * @brief Get the fifth radiation integral I5 for eigenmode U.
     * @return double Radiation integral I5 for eigenmode U
     */
    double I5u() const { return I5u_; }
    /**
     * @brief Get the fifth radiation integral I5 for eigenmode V.
     * @return double Radiation integral I5 for eigenmode V
     */
    double I5v() const { return I5v_; }
    /**
     * @brief Get the fourth radiation integral I4 for eigenmode U.
     * @return double Radiation integral I4 for eigenmode U
     */
    double I4u() const { return I4u_; }
    /**
     * @brief Get the fourth radiation integral I4 for eigenmode V.
     * @return double Radiation integral I4 for eigenmode V
     */
    double I4v() const { return I4v_; }

    // Update the ring parameters (tunes, etc.)
    void update(double delta=0.0,
        IntegrationMethod method=IntegrationMethod::MIDPOINT) noexcept(false);

    // find initial coordinates of closed orbit
    Coordinate find_initial_coordinate_of_closed_orbit(const Coordinate &cood_guess,
        IntegrationMethod method=IntegrationMethod::MIDPOINT) const noexcept(false);

    // Polymorphic clone
    std::shared_ptr<Element> clone() const noexcept(false) override;
};
