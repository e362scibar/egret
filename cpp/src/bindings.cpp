/**
 * @file bindings.cpp
 * @brief Pybind11 bindings for Egret C++ classes.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// bindings.cpp
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

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include "egret/drift.hpp"
#include "egret/steering.hpp"
#include "egret/dipole.hpp"
#include "egret/quadrupole.hpp"
#include "egret/sextupole.hpp"
#include "egret/octupole.hpp"
#include "egret/lattice.hpp"
#include "egret/ring.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace egret {
    class PyBaseArray;
    class PyElement;
    class PyNonlinearMultipole;
}

/**
 * @brief Trampoline class of BaseArray for pybind11 polymorphism
 */
class egret::PyBaseArray : public egret::BaseArray,
    public py::trampoline_self_life_support {
public:
    // Inherit parent constructor
    using BaseArray::BaseArray;

    size_t index_from_s(double s) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            size_t, // return type
            BaseArray, // parent class
            index_from_s, // python function name
            s // arguments
        );
    }
};

/**
 * @brief Trampoline class of Element for pybind11 polymorphism
 */
class egret::PyElement : public egret::Element {
public:
    // Inherit parent constructor
    using Element::Element;

    Eigen::Matrix4d transfer_matrix(const std::optional<Coordinate> &cood0,
        double ds, IntegrationMethod method) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            Eigen::Matrix4d, // return type
            Element, // parent class
            transfer_matrix, // python function name
            cood0, ds, method // arguments
        );
    }

    // define return type tuple_of_tmat_array to prevent macro error
    using tuple_of_tmat_array = std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>;
    tuple_of_tmat_array
    transfer_matrix_array(const std::optional<Coordinate> &cood0,
        double ds, bool endpoint, IntegrationMethod method) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            tuple_of_tmat_array, // return type
            Element, // parent class
            transfer_matrix_array, // python function name
            cood0, ds, endpoint, method // arguments
        );
    }

    Eigen::Vector4d dispersion(const std::optional<Coordinate> &cood0, double ds,
        IntegrationMethod method) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            Eigen::Vector4d, // return type
            Element, // parent class
            dispersion, // python function name
            cood0, ds, method // arguments
        );
    }

    // define return type tuple_of_disp_array to prevent macro error
    using tuple_of_disp_array = std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>;
    tuple_of_disp_array
    dispersion_array(const std::optional<Coordinate> &cood0,
        double ds, bool endpoint, IntegrationMethod method) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            tuple_of_disp_array, // return type
            Element, // parent class
            dispersion_array, // python function name
            cood0, ds, endpoint, method // arguments
        );
    }

    // define return type tuple_of_beam_params to prevent macro error
    using tuple_of_beam_params = std::tuple<Coordinate, std::optional<Envelope>, std::optional<Dispersion>>;
    tuple_of_beam_params
    transfer(const Coordinate &cood0,
        const std::optional<Envelope> &evlp0,
        const std::optional<Dispersion> &disp0,
        double ds, IntegrationMethod method) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            tuple_of_beam_params, // return type
            Element, // parent class
            transfer, // python function name
            cood0, evlp0, disp0, ds, method // arguments
        );
    }

    // define return type tuple_of_beam_param_array to prevent macro error
    using tuple_of_beam_param_array = std::tuple<CoordinateArray, std::optional<EnvelopeArray>, std::optional<DispersionArray>>;
    tuple_of_beam_param_array
    transfer_array(const Coordinate &cood0,
        const std::optional<Envelope> &evlp0,
        const std::optional<Dispersion> &disp0,
        double ds, bool endpoint, IntegrationMethod method) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            tuple_of_beam_param_array, // return type
            Element, // parent class
            transfer_array, // python function name
            cood0, evlp0, disp0, ds, endpoint, method // arguments
        );
    }

    using tuple_of_rad_ints =  std::tuple<double, double, double, double, double, double>;
    tuple_of_rad_ints
    radiation_integrals(const Coordinate &cood0, const Envelope &evlp0,
        const Dispersion &disp0, double ds, IntegrationMethod method) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            tuple_of_rad_ints, // return type
            Element, // parent class
            radiation_integrals, // python function name
            cood0, evlp0, disp0, ds, method // arguments
        );
    }
};

/**
 * @brief Trampoline class of NonlinearMultipole for pybind11 polymorphism
 */
class egret::PyNonlinearMultipole : public egret::NonlinearMultipole {
public:
    // Inherit parent constructor
    using NonlinearMultipole::NonlinearMultipole;

    // define return type tuple_of_k to prevent macro error
    using tuple_of_k = std::tuple<std::complex<double>, std::complex<double>>;
    tuple_of_k get_k(const Coordinate &cood) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            tuple_of_k, // return type
            NonlinearMultipole, // parent class
            get_k, // python function name
            cood // arguments
        );
    }
};

PYBIND11_MODULE(cppegret, m) {
    m.doc() = "pybind11 bindings for egret";

    py::enum_<egret::Element::IntegrationMethod>(m, "IntegrationMethod")
        .value("MIDPOINT", egret::Element::MIDPOINT)
        .value("RK4", egret::Element::RK4)
        .value("SYMPLECTIC1", egret::Element::SYMPLECTIC1)
        .value("SYMPLECTIC2", egret::Element::SYMPLECTIC2)
        .value("SYMPLECTIC4", egret::Element::SYMPLECTIC4)
        .export_values();

    py::class_<egret::BaseArray, egret::PyBaseArray, py::smart_holder>(m, "BaseArray")
        .def(py::init<const Eigen::ArrayXd&>(), py::arg("s_array"))
        .def_property("s_array",
            static_cast<const Eigen::ArrayXd&(egret::BaseArray::*)() const>(&egret::BaseArray::s_array),
            static_cast<void(egret::BaseArray::*)(const Eigen::ArrayXd&)>(&egret::BaseArray::s_array))
        .def("size", &egret::BaseArray::size)
        .def("__len__", &egret::BaseArray::size)
        .def("ds", &egret::BaseArray::ds)
        .def("append", &egret::BaseArray::append, py::arg("other"))
        .def("index_from_s", &egret::BaseArray::index_from_s, py::arg("s"));

    py::class_<egret::Coordinate>(m, "Coordinate")
        .def(py::init<const Eigen::Vector4d&, double, double, double>(),
            py::arg("vector") = Eigen::Vector4d::Zero(),
            py::arg("s") = 0.0, py::arg("z") = 0.0, py::arg("delta") = 0.0)
        .def_property("vector",
            static_cast<const Eigen::Vector4d&(egret::Coordinate::*)() const>(&egret::Coordinate::vector),
            static_cast<void(egret::Coordinate::*)(const Eigen::Vector4d&)>(&egret::Coordinate::vector))
        .def_property("x",
            static_cast<double(egret::Coordinate::*)() const>(&egret::Coordinate::x),
            static_cast<void(egret::Coordinate::*)(double)>(&egret::Coordinate::x))
        .def_property("xp",
            static_cast<double(egret::Coordinate::*)() const>(&egret::Coordinate::xp),
            static_cast<void(egret::Coordinate::*)(double)>(&egret::Coordinate::xp))
        .def_property("y",
            static_cast<double(egret::Coordinate::*)() const>(&egret::Coordinate::y),
            static_cast<void(egret::Coordinate::*)(double)>(&egret::Coordinate::y))
        .def_property("yp",
            static_cast<double(egret::Coordinate::*)() const>(&egret::Coordinate::yp),
            static_cast<void(egret::Coordinate::*)(double)>(&egret::Coordinate::yp))
        .def_property("s",
            static_cast<double(egret::Coordinate::*)() const>(&egret::Coordinate::s),
            static_cast<void(egret::Coordinate::*)(double)>(&egret::Coordinate::s))
        .def_property("z",
            static_cast<double(egret::Coordinate::*)() const>(&egret::Coordinate::z),
            static_cast<void(egret::Coordinate::*)(double)>(&egret::Coordinate::z))
        .def_property("delta",
            static_cast<double(egret::Coordinate::*)() const>(&egret::Coordinate::delta),
            static_cast<void(egret::Coordinate::*)(double)>(&egret::Coordinate::delta));

    py::class_<egret::CoordinateArray, egret::BaseArray, py::smart_holder>(m, "CoordinateArray")
        .def(py::init<const Eigen::Matrix<double,4,Eigen::Dynamic>&,
            const Eigen::ArrayXd&, const Eigen::ArrayXd&, const Eigen::ArrayXd&>(),
            py::arg("vector_array"), py::arg("s_array"),
            py::arg("z_array") = Eigen::ArrayXd(), py::arg("delta_array") = Eigen::ArrayXd())
        .def_property("vector_array",
            static_cast<const Eigen::Matrix<double,4,Eigen::Dynamic>&(egret::CoordinateArray::*)() const>(&egret::CoordinateArray::vector_array),
            static_cast<void(egret::CoordinateArray::*)(const Eigen::Matrix<double,4,Eigen::Dynamic>&)>(&egret::CoordinateArray::vector_array))
        .def_property("z_array",
            static_cast<const Eigen::ArrayXd&(egret::CoordinateArray::*)() const>(&egret::CoordinateArray::z_array),
            static_cast<void(egret::CoordinateArray::*)(const Eigen::ArrayXd&)>(&egret::CoordinateArray::z_array))
        .def_property("delta_array",
            static_cast<const Eigen::ArrayXd&(egret::CoordinateArray::*)() const>(&egret::CoordinateArray::delta_array),
            static_cast<void(egret::CoordinateArray::*)(const Eigen::ArrayXd&)>(&egret::CoordinateArray::delta_array))
        .def_property("x_array",
            static_cast<Eigen::ArrayXd(egret::CoordinateArray::*)() const>(&egret::CoordinateArray::x_array),
            static_cast<void(egret::CoordinateArray::*)(const Eigen::ArrayXd&)>(&egret::CoordinateArray::x_array))
        .def_property("xp_array",
            static_cast<Eigen::ArrayXd(egret::CoordinateArray::*)() const>(&egret::CoordinateArray::xp_array),
            static_cast<void(egret::CoordinateArray::*)(const Eigen::ArrayXd&)>(&egret::CoordinateArray::xp_array))
        .def_property("y_array",
            static_cast<Eigen::ArrayXd(egret::CoordinateArray::*)() const>(&egret::CoordinateArray::y_array),
            static_cast<void(egret::CoordinateArray::*)(const Eigen::ArrayXd&)>(&egret::CoordinateArray::y_array))
        .def_property("yp_array",
            static_cast<Eigen::ArrayXd(egret::CoordinateArray::*)() const>(&egret::CoordinateArray::yp_array),
            static_cast<void(egret::CoordinateArray::*)(const Eigen::ArrayXd&)>(&egret::CoordinateArray::yp_array))
        .def("append", &egret::CoordinateArray::append, py::arg("other"))
        .def("from_s", &egret::CoordinateArray::from_s, py::arg("s"));

    py::class_<egret::Envelope>(m, "Envelope")
        .def(py::init<const Eigen::Matrix4d&, double, const std::optional<const Eigen::Matrix2d>&, double, double>(),
            py::arg("cov") = Eigen::Matrix4d::Identity(),
            py::arg("s") = 0.0, py::arg("T") = std::nullopt,
            py::arg("psix") = 0.0, py::arg("psiy") = 0.0)
        .def_property_readonly("cov", &egret::Envelope::cov)
        .def_property_readonly("s", &egret::Envelope::s)
        .def_property_readonly("T", &egret::Envelope::T)
        .def_property_readonly("tau", &egret::Envelope::tau)
        .def_property_readonly("U", &egret::Envelope::U)
        .def_property_readonly("V", &egret::Envelope::V)
        .def_property_readonly("psix", &egret::Envelope::psix)
        .def_property_readonly("psiy", &egret::Envelope::psiy)
        .def_property_readonly("bx", &egret::Envelope::bx)
        .def_property_readonly("ax", &egret::Envelope::ax)
        .def_property_readonly("gx", &egret::Envelope::gx)
        .def_property_readonly("by", &egret::Envelope::by)
        .def_property_readonly("ay", &egret::Envelope::ay)
        .def_property_readonly("gy", &egret::Envelope::gy)
        .def_property_readonly("bu", &egret::Envelope::bu)
        .def_property_readonly("au", &egret::Envelope::au)
        .def_property_readonly("gu", &egret::Envelope::gu)
        .def_property_readonly("bv", &egret::Envelope::bv)
        .def_property_readonly("av", &egret::Envelope::av)
        .def_property_readonly("gv", &egret::Envelope::gv)
        .def_property_readonly("T_matrix", &egret::Envelope::T_matrix)
        .def_static("adjoint", &egret::Envelope::adjoint, py::arg("M"))
        .def("calc_eigenmode", &egret::Envelope::calc_eigenmode, py::arg("T") = nullptr)
        .def("transfer", &egret::Envelope::transfer, py::arg("M"), py::arg("length"));

    py::class_<egret::EnvelopeArray, egret::BaseArray, py::smart_holder>(m, "EnvelopeArray")
        .def(py::init<const std::vector<Eigen::Matrix4d>&, const Eigen::ArrayXd&,
            const std::optional<std::vector<Eigen::Matrix2d>>&,
            const std::optional<Eigen::ArrayXd>&,
            const std::optional<Eigen::ArrayXd>&>(),
            py::arg("cov_array"), py::arg("s_array"), py::arg("T_array") = std::nullopt,
            py::arg("psix_array") = std::nullopt, py::arg("psiy_array") = std::nullopt)
        .def("cov", &egret::EnvelopeArray::cov, py::arg("index"))
        .def("T", &egret::EnvelopeArray::T, py::arg("index"))
        .def("tau", &egret::EnvelopeArray::tau, py::arg("index"))
        .def("U", &egret::EnvelopeArray::U, py::arg("index"))
        .def("V", &egret::EnvelopeArray::V, py::arg("index"))
        .def("psix", &egret::EnvelopeArray::psix, py::arg("index"))
        .def("psiy", &egret::EnvelopeArray::psiy, py::arg("index"))
        .def("bx", &egret::EnvelopeArray::bx, py::arg("index"))
        .def("ax", &egret::EnvelopeArray::ax, py::arg("index"))
        .def("gx", &egret::EnvelopeArray::gx, py::arg("index"))
        .def("by", &egret::EnvelopeArray::by, py::arg("index"))
        .def("ay", &egret::EnvelopeArray::ay, py::arg("index"))
        .def("gy", &egret::EnvelopeArray::gy, py::arg("index"))
        .def("bu", &egret::EnvelopeArray::bu, py::arg("index"))
        .def("au", &egret::EnvelopeArray::au, py::arg("index"))
        .def("gu", &egret::EnvelopeArray::gu, py::arg("index"))
        .def("bv", &egret::EnvelopeArray::bv, py::arg("index"))
        .def("av", &egret::EnvelopeArray::av, py::arg("index"))
        .def("gv", &egret::EnvelopeArray::gv, py::arg("index"))
        .def("T_matrix", &egret::EnvelopeArray::T_matrix, py::arg("index"))
        .def_property_readonly("cov_array", &egret::EnvelopeArray::cov_array)
        .def_property_readonly("T_array", &egret::EnvelopeArray::T_array)
        .def_property_readonly("tau_array", &egret::EnvelopeArray::tau_array)
        .def_property_readonly("U_array", &egret::EnvelopeArray::U_array)
        .def_property_readonly("V_array", &egret::EnvelopeArray::V_array)
        .def_property_readonly("psix_array", &egret::EnvelopeArray::psix_array)
        .def_property_readonly("psiy_array", &egret::EnvelopeArray::psiy_array)
        .def_property_readonly("bx_array", &egret::EnvelopeArray::bx_array)
        .def_property_readonly("ax_array", &egret::EnvelopeArray::ax_array)
        .def_property_readonly("gx_array", &egret::EnvelopeArray::gx_array)
        .def_property_readonly("by_array", &egret::EnvelopeArray::by_array)
        .def_property_readonly("ay_array", &egret::EnvelopeArray::ay_array)
        .def_property_readonly("gy_array", &egret::EnvelopeArray::gy_array)
        .def_property_readonly("bu_array", &egret::EnvelopeArray::bu_array)
        .def_property_readonly("au_array", &egret::EnvelopeArray::au_array)
        .def_property_readonly("gu_array", &egret::EnvelopeArray::gu_array)
        .def_property_readonly("bv_array", &egret::EnvelopeArray::bv_array)
        .def_property_readonly("av_array", &egret::EnvelopeArray::av_array)
        .def_property_readonly("gv_array", &egret::EnvelopeArray::gv_array)
        .def_property_readonly("T_matrix_array", &egret::EnvelopeArray::T_matrix_array)
        .def("calc_eigenmode", &egret::EnvelopeArray::calc_eigenmode,
            py::arg("T_array") = std::nullopt)
        .def("append", &egret::EnvelopeArray::append, py::arg("other"))
        .def("from_s", &egret::EnvelopeArray::from_s, py::arg("s"))
        .def("transport", &egret::EnvelopeArray::transport,
            py::arg("M_array"), py::arg("s_array"));

    py::class_<egret::Dispersion>(m, "Dispersion")
        .def(py::init<const Eigen::Vector4d&, double>(),
            py::arg("vector") = Eigen::Vector4d::Zero(),
            py::arg("s") = 0.0)
        .def_property("vector",
            static_cast<const Eigen::Vector4d&(egret::Dispersion::*)() const>(&egret::Dispersion::vector),
            static_cast<void(egret::Dispersion::*)(const Eigen::Vector4d&)>(&egret::Dispersion::vector))
        .def_property("s",
            static_cast<double(egret::Dispersion::*)() const>(&egret::Dispersion::s),
            static_cast<void(egret::Dispersion::*)(double)>(&egret::Dispersion::s))
        .def_property("x",
            static_cast<double(egret::Dispersion::*)() const>(&egret::Dispersion::x),
            static_cast<void(egret::Dispersion::*)(double)>(&egret::Dispersion::x))
        .def_property("xp",
            static_cast<double(egret::Dispersion::*)() const>(&egret::Dispersion::xp),
            static_cast<void(egret::Dispersion::*)(double)>(&egret::Dispersion::xp))
        .def_property("y",
            static_cast<double(egret::Dispersion::*)() const>(&egret::Dispersion::y),
            static_cast<void(egret::Dispersion::*)(double)>(&egret::Dispersion::y))
        .def_property("yp",
            static_cast<double(egret::Dispersion::*)() const>(&egret::Dispersion::yp),
            static_cast<void(egret::Dispersion::*)(double)>(&egret::Dispersion::yp));

    py::class_<egret::DispersionArray, egret::BaseArray, py::smart_holder>(m, "DispersionArray")
        .def(py::init<const Eigen::Matrix<double, 4, Eigen::Dynamic>&, const Eigen::ArrayXd&>(),
            py::arg("vector_array"), py::arg("s_array"))
        .def_property("vector_array",
            static_cast<const Eigen::Matrix<double, 4, Eigen::Dynamic>&(egret::DispersionArray::*)() const>(&egret::DispersionArray::vector_array),
            static_cast<void(egret::DispersionArray::*)(const Eigen::Matrix<double, 4, Eigen::Dynamic>&)>(&egret::DispersionArray::vector_array))
        .def_property("x_array",
            static_cast<Eigen::ArrayXd(egret::DispersionArray::*)() const>(&egret::DispersionArray::x_array),
            static_cast<void(egret::DispersionArray::*)(const Eigen::ArrayXd&)>(&egret::DispersionArray::x_array))
        .def_property("xp_array",
            static_cast<Eigen::ArrayXd(egret::DispersionArray::*)() const>(&egret::DispersionArray::xp_array),
            static_cast<void(egret::DispersionArray::*)(const Eigen::ArrayXd&)>(&egret::DispersionArray::xp_array))
        .def_property("y_array",
            static_cast<Eigen::ArrayXd(egret::DispersionArray::*)() const>(&egret::DispersionArray::y_array),
            static_cast<void(egret::DispersionArray::*)(const Eigen::ArrayXd&)>(&egret::DispersionArray::y_array))
        .def_property("yp_array",
            static_cast<Eigen::ArrayXd(egret::DispersionArray::*)() const>(&egret::DispersionArray::yp_array),
            static_cast<void(egret::DispersionArray::*)(const Eigen::ArrayXd&)>(&egret::DispersionArray::yp_array))
        .def("append", &egret::DispersionArray::append, py::arg("other"))
        .def("from_s", &egret::DispersionArray::from_s, py::arg("s"));

    py::class_<egret::Object, std::shared_ptr<egret::Object>>(m, "Object")
        .def(py::init<const std::string&>(), py::arg("name"))
        .def_property("name",
            static_cast<const std::string&(egret::Object::*)() const>(&egret::Object::name),
            static_cast<void(egret::Object::*)(const std::string&)>(&egret::Object::name));

    py::class_<egret::Element, egret::PyElement, egret::Object,
        std::shared_ptr<egret::Element>>(m, "Element")
        .def(py::init<const std::string&, double, double, double, double, double, double, const std::string&>(),
            py::arg("name"), py::arg("length"), py::arg("angle") = 0.0,
            py::arg("dx") = 0.0, py::arg("dy") = 0.0, py::arg("ds") = 0.0,
            py::arg("tilt") = 0.0, py::arg("info") = "")
        .def_property("length",
            static_cast<double(egret::Element::*)() const>(&egret::Element::length),
            static_cast<void(egret::Element::*)(double)>(&egret::Element::length))
        .def_property("angle",
            static_cast<double(egret::Element::*)() const>(&egret::Element::angle),
            static_cast<void(egret::Element::*)(double)>(&egret::Element::angle))
        .def_property("dx",
            static_cast<double(egret::Element::*)() const>(&egret::Element::dx),
            static_cast<void(egret::Element::*)(double)>(&egret::Element::dx))
        .def_property("dy",
            static_cast<double(egret::Element::*)() const>(&egret::Element::dy),
            static_cast<void(egret::Element::*)(double)>(&egret::Element::dy))
        .def_property("ds",
            static_cast<double(egret::Element::*)() const>(&egret::Element::ds),
            static_cast<void(egret::Element::*)(double)>(&egret::Element::ds))
        .def_property("tilt",
            static_cast<double(egret::Element::*)() const>(&egret::Element::tilt),
            static_cast<void(egret::Element::*)(double)>(&egret::Element::tilt))
        .def_property("info",
            static_cast<const std::string&(egret::Element::*)() const>(&egret::Element::info),
            static_cast<void(egret::Element::*)(const std::string&)>(&egret::Element::info))
        .def_property_readonly("elements", &egret::Element::elements)
        .def_property_readonly("indices", &egret::Element::get_indices)
        .def("s_array",
            static_cast<Eigen::ArrayXd(egret::Element::*)(double, bool) const>(&egret::Element::s_array),
            py::arg("ds") = 0.1, py::arg("endpoint") = true)
        .def("transfer_matrix", &egret::Element::transfer_matrix,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1,
                py::arg_v("method", egret::Element::SYMPLECTIC4, "SYMPLECTIC4"))
        .def("transfer_matrix_array", &egret::Element::transfer_matrix_array,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1, py::arg("endpoint") = false,
                py::arg_v("method", egret::Element::SYMPLECTIC4, "SYMPLECTIC4"))
        .def("dispersion", &egret::Element::dispersion,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1,
                py::arg_v("method", egret::Element::SYMPLECTIC4, "SYMPLECTIC4"))
        .def("dispersion_array", &egret::Element::dispersion_array,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1, py::arg("endpoint") = false,
                py::arg_v("method", egret::Element::SYMPLECTIC4, "SYMPLECTIC4"))
        .def("transfer", &egret::Element::transfer,
            py::arg("cood0"), py::arg("evlp0") = std::nullopt,
            py::arg("disp0") = std::nullopt, py::arg("ds") = 0.1,
                py::arg_v("method", egret::Element::SYMPLECTIC4, "SYMPLECTIC4"))
        .def("transfer_array", &egret::Element::transfer_array,
            py::arg("cood0"), py::arg("evlp0") = std::nullopt,
            py::arg("disp0") = std::nullopt, py::arg("ds") = 0.1,
            py::arg("endpoint") = false,
                py::arg_v("method", egret::Element::SYMPLECTIC4, "SYMPLECTIC4"))
        .def("radiation_integrals", &egret::Element::radiation_integrals,
            py::arg("cood0"), py::arg("evlp0"), py::arg("disp0"), py::arg("ds") = 0.1,
                py::arg_v("method", egret::Element::SYMPLECTIC4, "SYMPLECTIC4"))
        .def("get_element_from_s", &egret::Element::get_element_from_s, py::arg("s"))
        .def("transfer_matrix_from_s", &egret::Element::transfer_matrix_from_s,
            py::arg("s0"), py::arg("cood0"), py::arg("ds") = 0.1,
                py::arg_v("method", egret::Element::SYMPLECTIC4, "SYMPLECTIC4"))
        .def("get_element", &egret::Element::get_element, py::arg("indices"))
        .def("set_element", &egret::Element::set_element,
            py::arg("indices"), py::arg("element"))
        .def("get_s", &egret::Element::get_s, py::arg("indices"))
        .def("find_index", &egret::Element::find_index, py::arg("names"))
        .def("set_indices", &egret::Element::set_indices,
            py::arg("indices") = std::vector<size_t>());

    py::class_<egret::Drift, egret::Element, egret::Object,
        std::shared_ptr<egret::Drift>>(m, "Drift")
        .def(py::init<const std::string&, double, double, double, double, double, const std::string&>(),
            py::arg("name"), py::arg("length"),
            py::arg("dx") = 0.0, py::arg("dy") = 0.0, py::arg("ds") = 0.0,
            py::arg("tilt") = 0.0, py::arg("info") = "")
        .def_static("transfer_matrix_from_length",
            py::overload_cast<double>(&egret::Drift::transfer_matrix), py::arg("length"))
        .def_static("transfer_matrix_array_from_length",
            py::overload_cast<double, double, bool>(&egret::Drift::transfer_matrix_array),
            py::arg("length"), py::arg("ds") = 0.1, py::arg("endpoint") = false);

    py::class_<egret::Steering, egret::Element, egret::Object,
        std::shared_ptr<egret::Steering>>(m, "Steering")
        .def(py::init<const std::string&, double, double, double, double, double,
            double, double, const std::string&>(),
            py::arg("name"), py::arg("length"),
            py::arg("kick_x") = 0.0, py::arg("kick_y") = 0.0,
            py::arg("dx") = 0.0, py::arg("dy") = 0.0, py::arg("ds") = 0.0,
            py::arg("tilt") = 0.0, py::arg("info") = "")
        .def_property("kick_x",
            static_cast<double(egret::Steering::*)() const>(&egret::Steering::kick_x),
            static_cast<void(egret::Steering::*)(double)>(&egret::Steering::kick_x))
        .def_property("kick_y",
            static_cast<double(egret::Steering::*)() const>(&egret::Steering::kick_y),
            static_cast<void(egret::Steering::*)(double)>(&egret::Steering::kick_y))
        .def_property("kick",
            static_cast<std::tuple<double,double>(egret::Steering::*)() const>(&egret::Steering::kick),
            static_cast<void(egret::Steering::*)(double,double)>(&egret::Steering::kick))
        .def("set_steering", &egret::Steering::set_steering,
            py::arg("kick_x"), py::arg("kick_y"));

    py::class_<egret::Dipole, egret::Element, egret::Object,
        std::shared_ptr<egret::Dipole>>(m, "Dipole")
        .def(py::init<const std::string&, double, double, double, double, double,
            double, double, double, double, double, double, const std::string&>(),
            py::arg("name"), py::arg("length"), py::arg("angle"),
            py::arg("k1") = 0.0, py::arg("e1") = 0.0, py::arg("e2") = 0.0,
            py::arg("h1") = 0.0, py::arg("h2") = 0.0,
            py::arg("dx") = 0.0, py::arg("dy") = 0.0, py::arg("ds") = 0.0,
            py::arg("tilt") = 0.0, py::arg("info") = "")
        .def_property("k1",
            static_cast<double(egret::Dipole::*)() const>(&egret::Dipole::k1),
            static_cast<void(egret::Dipole::*)(double)>(&egret::Dipole::k1))
        .def_property("e1",
            static_cast<double(egret::Dipole::*)() const>(&egret::Dipole::e1),
            static_cast<void(egret::Dipole::*)(double)>(&egret::Dipole::e1))
        .def_property("e2",
            static_cast<double(egret::Dipole::*)() const>(&egret::Dipole::e2),
            static_cast<void(egret::Dipole::*)(double)>(&egret::Dipole::e2))
        .def_property("h1",
            static_cast<double(egret::Dipole::*)() const>(&egret::Dipole::h1),
            static_cast<void(egret::Dipole::*)(double)>(&egret::Dipole::h1))
        .def_property("h2",
            static_cast<double(egret::Dipole::*)() const>(&egret::Dipole::h2),
            static_cast<void(egret::Dipole::*)(double)>(&egret::Dipole::h2))
        .def_property_readonly("rho", &egret::Dipole::rho);

    py::class_<egret::Quadrupole, egret::Element, egret::Object,
        std::shared_ptr<egret::Quadrupole>>(m, "Quadrupole")
        .def(py::init<const std::string&, double, double, double, double, double, double, const std::string&>(),
            py::arg("name"), py::arg("length"), py::arg("k1"),
            py::arg("dx") = 0.0, py::arg("dy") = 0.0, py::arg("ds") = 0.0,
            py::arg("tilt") = 0.0, py::arg("info") = "")
        .def_property("k1",
            static_cast<double(egret::Quadrupole::*)() const>(&egret::Quadrupole::k1),
            static_cast<void(egret::Quadrupole::*)(double)>(&egret::Quadrupole::k1))
        .def_static("transfer_matrix_from_length_k1",
            py::overload_cast<double, double, const std::optional<Eigen::Matrix4d>&>
            (&egret::Quadrupole::transfer_matrix),
            py::arg("length"), py::arg("k1"), py::arg("rmat") = std::nullopt)
        .def_static("rotation_matrix", &egret::Quadrupole::rotation_matrix,
            py::arg("tilt"))
        .def_static("transfer_by_length_k1", &egret::Quadrupole::transfer,
            py::arg("cood0_vec"), py::arg("length"), py::arg("k1"),
            py::arg("k0x"), py::arg("k0y"), py::arg("tilt"),
            py::arg("tmat_flag") = false, py::arg("disp_flag") = false);

    py::class_<egret::NonlinearMultipole, egret::PyNonlinearMultipole, egret::Element, egret::Object,
    std::shared_ptr<egret::NonlinearMultipole>>(m, "NonlinearMultipole")
        .def(py::init<const std::string&, double, double, double,
            double, double, double, double, const std::string&>(),
            py::arg("name"), py::arg("length"),
            py::arg("kick_x") = 0.0, py::arg("kick_y") = 0.0,
            py::arg("dx") = 0.0, py::arg("dy") = 0.0, py::arg("ds") = 0.0,
            py::arg("tilt") = 0.0, py::arg("info") = "")
        .def_property("k0x",
            static_cast<double(egret::NonlinearMultipole::*)() const>(&egret::NonlinearMultipole::k0x),
            static_cast<void(egret::NonlinearMultipole::*)(double)>(&egret::NonlinearMultipole::k0x))
        .def_property("k0y",
            static_cast<double(egret::NonlinearMultipole::*)() const>(&egret::NonlinearMultipole::k0y),
            static_cast<void(egret::NonlinearMultipole::*)(double)>(&egret::NonlinearMultipole::k0y))
        .def_property("kick_x",
            static_cast<double(egret::NonlinearMultipole::*)() const>(&egret::NonlinearMultipole::kick_x),
            static_cast<void(egret::NonlinearMultipole::*)(double)>(&egret::NonlinearMultipole::kick_x))
        .def_property("kick_y",
            static_cast<double(egret::NonlinearMultipole::*)() const>(&egret::NonlinearMultipole::kick_y),
            static_cast<void(egret::NonlinearMultipole::*)(double)>(&egret::NonlinearMultipole::kick_y))
        .def("set_steering", &egret::NonlinearMultipole::set_steering,
            py::arg("kick_x"), py::arg("kick_y"))
        .def("get_k", &egret::NonlinearMultipole::get_k,
            py::arg("cood"));

    py::class_<egret::Sextupole, egret::NonlinearMultipole, egret::Element,
        egret::Object, std::shared_ptr<egret::Sextupole>>(m, "Sextupole")
        .def(py::init<const std::string&, double, double, double, double, double,
            double, double, double, const std::string&>(),
            py::arg("name"), py::arg("length"), py::arg("k2"),
            py::arg("kick_x") = 0.0, py::arg("kick_y") = 0.0,
            py::arg("dx") = 0.0, py::arg("dy") = 0.0, py::arg("ds") = 0.0,
            py::arg("tilt") = 0.0, py::arg("info") = "")
        .def_property("k2",
            static_cast<double(egret::Sextupole::*)() const>(&egret::Sextupole::k2),
            static_cast<void(egret::Sextupole::*)(double)>(&egret::Sextupole::k2));

    py::class_<egret::Octupole, egret::NonlinearMultipole, egret::Element,
        egret::Object, std::shared_ptr<egret::Octupole>>(m, "Octupole")
        .def(py::init<const std::string&, double, double, double, double, double,
            double, double, double, double, double, const std::string&>(),
            py::arg("name"), py::arg("length"), py::arg("k3"),
            py::arg("k1") = 0.0, py::arg("tilt_quad") = 0.0,
            py::arg("kick_x") = 0.0, py::arg("kick_y") = 0.0,
            py::arg("dx") = 0.0, py::arg("dy") = 0.0, py::arg("ds") = 0.0,
            py::arg("tilt") = 0.0, py::arg("info") = "")
        .def_property("k3",
            static_cast<double(egret::Octupole::*)() const>(&egret::Octupole::k3),
            static_cast<void(egret::Octupole::*)(double)>(&egret::Octupole::k3))
        .def_property("k1",
            static_cast<double(egret::Octupole::*)() const>(&egret::Octupole::k1),
            static_cast<void(egret::Octupole::*)(double)>(&egret::Octupole::k1))
        .def_property("tilt_quad",
            static_cast<double(egret::Octupole::*)() const>(&egret::Octupole::tilt_quad),
            static_cast<void(egret::Octupole::*)(double)>(&egret::Octupole::tilt_quad))
        .def("set_quadrupole", &egret::Octupole::set_quadrupole,
            py::arg("k1") = std::nullopt, py::arg("tilt_quad") = std::nullopt);

    py::class_<egret::Lattice, egret::Element, egret::Object,
        std::shared_ptr<egret::Lattice>>(m, "Lattice")
        .def(py::init<const std::string&, const std::vector<std::shared_ptr<egret::Element>>&,
            double, double, double, double, const std::string&>(),
        py::arg("name"), py::arg("elements"),
        py::arg("dx") = 0.0, py::arg("dy") = 0.0, py::arg("ds") = 0.0,
        py::arg("tilt") = 0.0, py::arg("info") = "")
        .def_static("length_of", &egret::Lattice::length,
            py::arg("elements"))
        .def_static("angle_of", &egret::Lattice::angle,
            py::arg("elements"));

    py::class_<egret::Ring, egret::Element, egret::Object, std::shared_ptr<egret::Ring>>(m, "Ring")
        .def(py::init<const std::string&, const std::vector<std::shared_ptr<egret::Element>>&,
            double, std::string&>(),
        py::arg("name"), py::arg("elements"),
        py::arg("energy"), py::arg("info") = "")
        .def_property_readonly("energy", &egret::Ring::energy)
        .def_property_readonly("tune_x", &egret::Ring::tune_x)
        .def_property_readonly("tune_y", &egret::Ring::tune_y)
        .def_property_readonly("cood0", &egret::Ring::cood0)
        .def_property_readonly("evlp0", &egret::Ring::evlp0)
        .def_property_readonly("disp0", &egret::Ring::disp0)
        .def_property_readonly("emittance_x", &egret::Ring::emittance_x)
        .def_property_readonly("emittance_y", &egret::Ring::emittance_y)
        .def_property_readonly("Jx", &egret::Ring::Jx)
        .def_property_readonly("Jy", &egret::Ring::Jy)
        .def_property_readonly("Jz", &egret::Ring::Jz)
        .def_property_readonly("I2", &egret::Ring::I2)
        .def_property_readonly("I4", &egret::Ring::I4)
        .def_property_readonly("I5u", &egret::Ring::I5u)
        .def_property_readonly("I5v", &egret::Ring::I5v)
        .def_property_readonly("I4u", &egret::Ring::I4u)
        .def_property_readonly("I4v", &egret::Ring::I4v)
        .def("update", &egret::Ring::update, py::arg("delta") = 0.0,
            py::arg_v("method", egret::Element::SYMPLECTIC4, "SYMPLECTIC4"))
        .def("find_initial_coordinate_of_closed_orbit",
            &egret::Ring::find_initial_coordinate_of_closed_orbit,
            py::arg("cood_guess") = std::nullopt,
            py::arg_v("method", egret::Element::SYMPLECTIC4, "SYMPLECTIC4"))
        .def_readonly_static("C_q", &egret::Ring::C_q)
        .def_readonly_static("m_e_eV", &egret::Ring::m_e_eV)
        .def_readwrite_static("tol_cod", &egret::Ring::tol_cod)
        .def_readwrite_static("max_iter_cod", &egret::Ring::max_iter_cod);
}
