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

#if 0
// Thin C++ wrapper types to provide Python-friendly element objects
// that store element parameters and delegate to existing static C++ APIs.
// These wrappers are local to bindings and are only intended to make
// the pybind11 interface more natural for Python callers.
struct QuadrupoleWrapper : public egret::Element {
    double length;
    double k1;
    double tilt;
    QuadrupoleWrapper(double length_, double k1_, double tilt_ = 0.0)
        : length(length_), k1(k1_), tilt(tilt_) {}
    // implement Element virtuals
    py::object transfer(const egret::Coordinate &cood0, double ds) override {
        (void) ds;
        Eigen::Matrix4d t = egret::Quadrupole::transfer_matrix(length, k1, tilt, cood0.delta);
        Eigen::Vector4d v = t * cood0.vector;
        egret::Coordinate out(v, cood0.s + length, cood0.z, cood0.delta);
        return py::make_tuple(out, py::none(), py::none());
    }
    py::object transfer_array(const egret::Coordinate &cood0, double ds, bool endpoint) override {
        auto pr = egret::Quadrupole::transfer_matrix_array(length, k1, tilt, cood0.delta, ds, endpoint);
        auto &tensor = pr.first; auto &svec = pr.second;
        int n = static_cast<int>(tensor.dimension(2));
        py::array_t<double> arr({4,4,n});
        auto buf = arr.mutable_unchecked<3>();
        for (int k=0;k<n;++k) for (int i=0;i<4;++i) for (int j=0;j<4;++j) buf(i,j,k) = tensor(i,j,k);
        py::array_t<double> sarr({n});
        auto sbuf = sarr.mutable_unchecked<1>();
        for (int i=0;i<n;++i) sbuf(i) = svec[i];
        return py::make_tuple(arr, sarr);
    }
};

struct DipoleWrapper : public egret::Element {
    double length;
    double angle;
    double k1;
    DipoleWrapper(double length_, double angle_, double k1_ = 0.0)
        : length(length_), angle(angle_), k1(k1_) {}
    py::object transfer(const egret::Coordinate &cood0, double ds) override {
        (void) ds;
        Eigen::Matrix4d t = egret::Dipole::transfer_matrix(length, angle, k1, cood0.delta);
        Eigen::Vector4d v = t * cood0.vector;
        egret::Coordinate out(v, cood0.s + length, cood0.z, cood0.delta);
        return py::make_tuple(out, py::none(), py::none());
    }
    py::object transfer_array(const egret::Coordinate &cood0, double ds, bool endpoint) override {
        auto pr = egret::Dipole::transfer_matrix_array(length, angle, k1, cood0.delta, ds, endpoint);
        auto &tensor = pr.first; auto &svec = pr.second;
        int n = static_cast<int>(tensor.dimension(2));
        py::array_t<double> arr({4,4,n});
        auto buf = arr.mutable_unchecked<3>();
        for (int k=0;k<n;++k) for (int i=0;i<4;++i) for (int j=0;j<4;++j) buf(i,j,k) = tensor(i,j,k);
        py::array_t<double> sarr({n});
        auto sbuf = sarr.mutable_unchecked<1>();
        for (int i=0;i<n;++i) sbuf(i) = svec[i];
        return py::make_tuple(arr, sarr);
    }
};

struct SextupoleWrapper : public egret::Element {
    double length;
    double k2;
    double dx;
    double dy;
    double ds;
    SextupoleWrapper(double length_, double k2_, double dx_ = 0.0, double dy_ = 0.0, double ds_ = 0.1)
        : length(length_), k2(k2_), dx(dx_), dy(dy_), ds(ds_) {}
    py::object transfer(const egret::Coordinate &cood0, double ds_in) override {
        Eigen::Matrix4d t = egret::Sextupole::transfer_matrix(cood0, length, k2, ds_in);
        Eigen::Vector4d v = t * cood0.vector;
        egret::Coordinate out(v, cood0.s + length, cood0.z, cood0.delta);
        return py::make_tuple(out, py::none(), py::none());
    }
    py::object transfer_array(const egret::Coordinate &cood0, double ds_in, bool endpoint) override {
        auto pr = egret::Sextupole::transfer_matrix_array(cood0, length, k2, ds_in, endpoint);
        auto &tensor = pr.first; auto &svec = pr.second;
        int n = static_cast<int>(tensor.dimension(2));
        py::array_t<double> arr({4,4,n});
        auto buf = arr.mutable_unchecked<3>();
        for (int k=0;k<n;++k) for (int i=0;i<4;++i) for (int j=0;j<4;++j) buf(i,j,k) = tensor(i,j,k);
        py::array_t<double> sarr({n});
        auto sbuf = sarr.mutable_unchecked<1>();
        for (int i=0;i<n;++i) sbuf(i) = svec[i];
        return py::make_tuple(arr, sarr);
    }
};
#endif

namespace egret {
    class PyBaseArray;
    class PyObject;
    class PyElement;
    class PyNonlinearMultipole;
}

/**
 * @brief Trampoline class of BaseArray for pybind11 polymorphism
 */
class egret::PyBaseArray : public egret::BaseArray,
    public pybind11::trampoline_self_life_support {
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

#if 0
/**
 * @brief Trampoline class of Object for pybind11 polymorphism
 */
class egret::PyObject : public egret::Object,
    public pybind11::trampoline_self_life_support {
public:
    // Inherit parent constructor
    using Object::Object;
};
#endif

/**
 * @brief Trampoline class of Element for pybind11 polymorphism
 */
class egret::PyElement : public egret::Element {
public:
    // Inherit parent constructor
    using Element::Element;

    Eigen::Matrix4d transfer_matrix(const std::optional<Coordinate> &cood0,
        double ds) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            Eigen::Matrix4d, // return type
            Element, // parent class
            transfer_matrix, // python function name
            cood0, ds // arguments
        );
    }

    // define return type tuple_of_tmat_array to prevent macro error
    using tuple_of_tmat_array = std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>;
    tuple_of_tmat_array
    transfer_matrix_array(const std::optional<Coordinate> &cood0,
        double ds, bool endpoint) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            tuple_of_tmat_array, // return type
            Element, // parent class
            transfer_matrix_array, // python function name
            cood0, ds, endpoint // arguments
        );
    }

    Eigen::Vector4d dispersion(const std::optional<Coordinate> &cood0, double ds) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            Eigen::Vector4d, // return type
            Element, // parent class
            dispersion, // python function name
            cood0, ds // arguments
        );
    }

    // define return type tuple_of_disp_array to prevent macro error
    using tuple_of_disp_array = std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>;
    tuple_of_disp_array
    dispersion_array(const std::optional<Coordinate> &cood0,
        double ds, bool endpoint) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            tuple_of_disp_array, // return type
            Element, // parent class
            dispersion_array, // python function name
            cood0, ds, endpoint // arguments
        );
    }

    // define return type tuple_of_beam_params to prevent macro error
    using tuple_of_beam_params = std::tuple<Coordinate, std::optional<Envelope>, std::optional<Dispersion>>;
    tuple_of_beam_params
    transfer(const Coordinate &cood0,
        const std::optional<Envelope> &evlp0,
        const std::optional<Dispersion> &disp0,
        double ds) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            tuple_of_beam_params, // return type
            Element, // parent class
            transfer, // python function name
            cood0, evlp0, disp0, ds // arguments
        );
    }

    // define return type tuple_of_beam_param_array to prevent macro error
    using tuple_of_beam_param_array = std::tuple<CoordinateArray, std::optional<EnvelopeArray>, std::optional<DispersionArray>>;
    tuple_of_beam_param_array
    transfer_array(const Coordinate &cood0,
        const std::optional<Envelope> &evlp0,
        const std::optional<Dispersion> &disp0,
        double ds, bool endpoint) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            tuple_of_beam_param_array, // return type
            Element, // parent class
            transfer_array, // python function name
            cood0, evlp0, disp0, ds, endpoint // arguments
        );
    }

    using tuple_of_rad_ints =  std::tuple<double, double, double, double, double, double>;
    tuple_of_rad_ints
    radiation_integrals(const Coordinate &cood0, const Envelope &evlp0,
        const Dispersion &disp0, double ds) const noexcept(false) override {
        PYBIND11_OVERRIDE(
            tuple_of_rad_ints, // return type
            Element, // parent class
            radiation_integrals, // python function name
            cood0, evlp0, disp0, ds // arguments
        );
    }
};

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

PYBIND11_MODULE(pyegret, m) {
    m.doc() = "pybind11 bindings for egret";

    py::class_<egret::BaseArray, egret::PyBaseArray, py::smart_holder>(m, "BaseArray")
        .def(py::init<const Eigen::ArrayXd&>(), py::arg("s_array"))
        .def_property_readonly("size", &egret::BaseArray::size)
        .def_property("s_array",
            static_cast<const Eigen::ArrayXd&(egret::BaseArray::*)() const>(&egret::BaseArray::s_array),
            static_cast<void(egret::BaseArray::*)(const Eigen::ArrayXd&)>(&egret::BaseArray::s_array))
        .def("ds", &egret::BaseArray::ds)
        .def("append", &egret::BaseArray::append, py::arg("other"))
        .def("index_from_s", &egret::BaseArray::index_from_s, py::arg("s"));

    py::class_<egret::Coordinate>(m, "Coordinate")
        .def(py::init<const Eigen::Vector4d&, double, double, double>(),
             py::arg("vector") = Eigen::Vector4d::Zero(),
             py::arg("s") = 0.0,
             py::arg("z") = 0.0,
             py::arg("delta") = 0.0)
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
            py::arg("vector_array"),
            py::arg("s_array"),
            py::arg("z_array") = Eigen::ArrayXd(),
            py::arg("delta_array") = Eigen::ArrayXd())
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
            static_cast<const Eigen::ArrayXd(egret::CoordinateArray::*)() const>(&egret::CoordinateArray::x_array),
            static_cast<void(egret::CoordinateArray::*)(const Eigen::ArrayXd&)>(&egret::CoordinateArray::x_array))
        .def_property("xp_array",
            static_cast<const Eigen::ArrayXd(egret::CoordinateArray::*)() const>(&egret::CoordinateArray::xp_array),
            static_cast<void(egret::CoordinateArray::*)(const Eigen::ArrayXd&)>(&egret::CoordinateArray::xp_array))
        .def_property("y_array",
            static_cast<const Eigen::ArrayXd(egret::CoordinateArray::*)() const>(&egret::CoordinateArray::y_array),
            static_cast<void(egret::CoordinateArray::*)(const Eigen::ArrayXd&)>(&egret::CoordinateArray::y_array))
        .def_property("yp_array",
            static_cast<const Eigen::ArrayXd(egret::CoordinateArray::*)() const>(&egret::CoordinateArray::yp_array),
            static_cast<void(egret::CoordinateArray::*)(const Eigen::ArrayXd&)>(&egret::CoordinateArray::yp_array))
        .def("append", &egret::CoordinateArray::append, py::arg("other"))
        .def("from_s", &egret::CoordinateArray::from_s, py::arg("s"));

    py::class_<egret::Envelope>(m, "Envelope")
        .def(py::init<const Eigen::Matrix4d&, double, const std::optional<const Eigen::Matrix2d>&>(),
            py::arg("cov") = Eigen::Matrix4d::Identity(),
            py::arg("s") = 0.0,
            py::arg("T") = std::nullopt)
        .def_property_readonly("cov", &egret::Envelope::cov)
        .def_property_readonly("s", &egret::Envelope::s)
        .def_property_readonly("T", &egret::Envelope::T)
        .def_property_readonly("tau", &egret::Envelope::tau)
        .def_property_readonly("U", &egret::Envelope::U)
        .def_property_readonly("V", &egret::Envelope::V)
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
        .def("transfer", &egret::Envelope::transfer, py::arg("M"), py::arg("length"));

    py::class_<egret::EnvelopeArray, egret::BaseArray, py::smart_holder>(m, "EnvelopeArray")
        .def(py::init<const std::vector<Eigen::Matrix4d>&, const Eigen::ArrayXd&,
            const std::optional<std::vector<Eigen::Matrix2d>>&>(),
            py::arg("cov_array"), py::arg("s_array"), py::arg("T_array") = std::nullopt)
        .def("cov", &egret::EnvelopeArray::cov, py::arg("index"))
        .def("T", &egret::EnvelopeArray::T, py::arg("index"))
        .def("tau", &egret::EnvelopeArray::tau, py::arg("index"))
        .def("U", &egret::EnvelopeArray::U, py::arg("index"))
        .def("V", &egret::EnvelopeArray::V, py::arg("index"))
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
        .def("bx_array", &egret::EnvelopeArray::bx_array)
        .def("ax_array", &egret::EnvelopeArray::ax_array)
        .def("gx_array", &egret::EnvelopeArray::gx_array)
        .def("by_array", &egret::EnvelopeArray::by_array)
        .def("ay_array", &egret::EnvelopeArray::ay_array)
        .def("gy_array", &egret::EnvelopeArray::gy_array)
        .def("bu_array", &egret::EnvelopeArray::bu_array)
        .def("au_array", &egret::EnvelopeArray::au_array)
        .def("gu_array", &egret::EnvelopeArray::gu_array)
        .def("bv_array", &egret::EnvelopeArray::bv_array)
        .def("av_array", &egret::EnvelopeArray::av_array)
        .def("gv_array", &egret::EnvelopeArray::gv_array)
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
        .def_property_readonly("vector_array", &egret::DispersionArray::vector_array)
        .def_property_readonly("x_array", &egret::DispersionArray::x_array)
        .def_property_readonly("xp_array", &egret::DispersionArray::xp_array)
        .def_property_readonly("y_array", &egret::DispersionArray::y_array)
        .def_property_readonly("yp_array", &egret::DispersionArray::yp_array)
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
        .def("s_array",
            static_cast<Eigen::ArrayXd(egret::Element::*)(double, bool) const>(&egret::Element::s_array),
            py::arg("ds") = 0.1, py::arg("endpoint") = true)
        .def("transfer_matrix", &egret::Element::transfer_matrix,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1)
        .def("transfer_matrix_array", &egret::Element::transfer_matrix_array,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1, py::arg("endpoint") = false)
        .def("dispersion", &egret::Element::dispersion,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1)
        .def("dispersion_array", &egret::Element::dispersion_array,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1, py::arg("endpoint") = false)
        .def("transfer", &egret::Element::transfer,
            py::arg("cood0"), py::arg("evlp0") = std::nullopt,
            py::arg("disp0") = std::nullopt, py::arg("ds") = 0.1)
        .def("transfer_array", &egret::Element::transfer_array,
            py::arg("cood0"), py::arg("evlp0") = std::nullopt,
            py::arg("disp0") = std::nullopt, py::arg("ds") = 0.1,
            py::arg("endpoint") = false)
        .def("radiation_integrals", &egret::Element::radiation_integrals,
            py::arg("cood0"), py::arg("evlp0"), py::arg("disp0"), py::arg("ds") = 0.1)
        .def("get_element_from_s", &egret::Element::get_element_from_s, py::arg("s"))
        .def("transfer_matrix_from_s", &egret::Element::transfer_matrix_from_s,
            py::arg("s0"), py::arg("cood0"), py::arg("ds") = 0.1)
        .def("get_element", &egret::Element::get_element, py::arg("indices"))
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
        /*
        .def("transfer_matrix",
            static_cast<Eigen::Matrix4d(egret::Drift::*)(const std::optional<egret::Coordinate>&, double) const>
            (&egret::Drift::transfer_matrix),
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1)
        .def("transfer_matrix_array",
            static_cast<std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
            (egret::Drift::*)(const std::optional<egret::Coordinate>&, double, bool) const>
            (&egret::Drift::transfer_matrix_array),
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1,
            py::arg("endpoint") = false
        */

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
        /*
        .def("transfer_matrix", &egret::Dipole::transfer_matrix,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1)
        .def("transfer_matrix_array", &egret::Dipole::transfer_matrix_array,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1, py::arg("endpoint") = false)
        .def("dispersion", &egret::Dipole::dispersion,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1)
        .def("dispersion_array", &egret::Dipole::dispersion_array,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1, py::arg("endpoint") = false)
        .def("transfer", &egret::Dipole::transfer,
            py::arg("cood0"), py::arg("evlp0") = std::nullopt,
            py::arg("disp0") = std::nullopt, py::arg("ds") = 0.1)
        .def("transfer_array", &egret::Dipole::transfer_array,
            py::arg("cood0"), py::arg("evlp0") = std::nullopt,
            py::arg("disp0") = std::nullopt, py::arg("ds") = 0.1,
            py::arg("endpoint") = false)
        .def("radiation_integrals", &egret::Dipole::radiation_integrals,
            py::arg("cood0"), py::arg("evlp0"), py::arg("disp0"), py::arg("ds") = 0.1);
        */

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
        /*
        .def("transfer_matrix",
            static_cast<Eigen::Matrix4d(egret::Quadrupole::*)(const std::optional<egret::Coordinate>&, double) const>
            (&egret::Quadrupole::transfer_matrix),
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1)
        .def("transfer_matrix_array", &egret::Quadrupole::transfer_matrix_array,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1, py::arg("endpoint") = false)
        .def_static("dispersion_from_length_k1",
            py::overload_cast<const Eigen::Vector4d&, double, double, double>
            (&egret::Quadrupole::dispersion),
            py::arg("cood0_vec"), py::arg("length"), py::arg("k1"), py::arg("tilt"))
        .def("dispersion",
            static_cast<Eigen::Vector4d(egret::Quadrupole::*)(const std::optional<egret::Coordinate>&, double) const>(&egret::Quadrupole::dispersion),
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1)
        .def("dispersion_array", &egret::Quadrupole::dispersion_array,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1, py::arg("endpoint") = false)
        */
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
        /*
        .def("transfer_by_midpoint_method", &egret::NonlinearMultipole::transfer_by_midpoint_method,
            py::arg("cood0"), py::arg("ds"),
            py::arg("tmat_flag") = true, py::arg("disp_flag") = false)
        .def("transfer_matrix", &egret::NonlinearMultipole::transfer_matrix,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1)
        .def("transfer_matrix_array", &egret::NonlinearMultipole::transfer_matrix_array,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1, py::arg("endpoint") = false)
        .def("dispersion", &egret::NonlinearMultipole::dispersion,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1)
        .def("dispersion_array", &egret::NonlinearMultipole::dispersion_array,
            py::arg("cood0") = std::nullopt, py::arg("ds") = 0.1, py::arg("endpoint") = false)
        .def("transfer", &egret::NonlinearMultipole::transfer,
            py::arg("cood0"), py::arg("evlp0") = std::nullopt,
            py::arg("disp0") = std::nullopt, py::arg("ds") = 0.1)
        .def("transfer_array", &egret::NonlinearMultipole::transfer_array,
            py::arg("cood0"), py::arg("evlp0") = std::nullopt,
            py::arg("disp0") = std::nullopt, py::arg("ds") = 0.1,
            py::arg("endpoint") = false);
        */

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
        //.def("get_k", &egret::Sextupole::get_k, py::arg("cood"));

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
        //.def("get_k", &egret::Octupole::get_k, py::arg("cood"));

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
        .def_property_readonly("J_x", &egret::Ring::J_x)
        .def_property_readonly("J_y", &egret::Ring::J_y)
        .def_property_readonly("J_z", &egret::Ring::J_z)
        .def("update", &egret::Ring::update, py::arg("delta") = 0.0)
        .def("find_initial_coordinate_of_closed_orbit",
            &egret::Ring::find_initial_coordinate_of_closed_orbit,
            py::arg("cood_guess") = std::nullopt)
        .def_readonly_static("C_q", &egret::Ring::C_q)
        .def_readonly_static("m_e_eV", &egret::Ring::m_e_eV)
        .def_readwrite_static("tol_cod", &egret::Ring::tol_cod)
        .def_readwrite_static("max_iter_cod", &egret::Ring::max_iter_cod);

#if 0
    // Simple container types for envelope and dispersion (numpy-backed)
    struct Envelope {
        py::array_t<double> beta_x;
        py::array_t<double> beta_y;
        py::array_t<double> s;
        Envelope(py::array_t<double> bx, py::array_t<double> by, py::array_t<double> ss)
            : beta_x(bx), beta_y(by), s(ss) {}
    };

    struct Dispersion {
        py::array_t<double> eta_x;
        py::array_t<double> eta_y;
        py::array_t<double> s;
        Dispersion(py::array_t<double> ex, py::array_t<double> ey, py::array_t<double> ss)
            : eta_x(ex), eta_y(ey), s(ss) {}
    };

    py::class_<Envelope>(m, "Envelope")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>>())
        .def_readwrite("beta_x", &Envelope::beta_x)
        .def_readwrite("beta_y", &Envelope::beta_y)
        .def_readwrite("s", &Envelope::s)
        .def(py::pickle(
            [](const Envelope &e) {
                return py::make_tuple(e.beta_x, e.beta_y, e.s);
            },
            [](py::tuple t) {
                if (t.size() != 3) throw std::runtime_error("Invalid Envelope pickle");
                return Envelope(t[0].cast<py::array_t<double>>(), t[1].cast<py::array_t<double>>(), t[2].cast<py::array_t<double>>());
            }
        ));

    py::class_<Dispersion>(m, "Dispersion")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>>())
        .def_readwrite("eta_x", &Dispersion::eta_x)
        .def_readwrite("eta_y", &Dispersion::eta_y)
        .def_readwrite("s", &Dispersion::s)
        .def(py::pickle(
            [](const Dispersion &d) {
                return py::make_tuple(d.eta_x, d.eta_y, d.s);
            },
            [](py::tuple t) {
                if (t.size() != 3) throw std::runtime_error("Invalid Dispersion pickle");
                return Dispersion(t[0].cast<py::array_t<double>>(), t[1].cast<py::array_t<double>>(), t[2].cast<py::array_t<double>>());
            }
        ));

    // Provide a lightweight trampoline class for egret::Element so Python can
    // subclass and C++ can call virtuals implemented in Python.
    struct PyElement : egret::Element {
        /* In the trampoline we use the pybind11 macros to forward calls */
        using egret::Element::Element;
        py::object transfer(const egret::Coordinate &cood0, double ds) override {
            PYBIND11_OVERRIDE_PURE(py::object, egret::Element, transfer, cood0, ds);
        }
        py::object transfer_array(const egret::Coordinate &cood0, double ds, bool endpoint) override {
            PYBIND11_OVERRIDE_PURE(py::object, egret::Element, transfer_array, cood0, ds, endpoint);
        }
    };

    // Bind Element base to allow C++ -> Python virtual dispatch
    py::class_<egret::Element, PyElement, std::shared_ptr<egret::Element>>(m, "Element")
        .def(py::init<>())
        .def("transfer", &egret::Element::transfer)
        .def("transfer_array", &egret::Element::transfer_array);

    // Bind thin wrapper classes for element objects (use shared_ptr and dynamic_attr to allow Python subclassing)
    py::class_<QuadrupoleWrapper, egret::Element, std::shared_ptr<QuadrupoleWrapper>>(m, "Quadrupole", py::dynamic_attr())
        .def(py::init<double, double, double>(), py::arg("length"), py::arg("k1"), py::arg("tilt") = 0.0)
        .def_readwrite("length", &QuadrupoleWrapper::length)
        .def_readwrite("k1", &QuadrupoleWrapper::k1)
        .def_readwrite("tilt", &QuadrupoleWrapper::tilt)
        .def("transfer_matrix",
             [] (const QuadrupoleWrapper &self, const egret::Coordinate &cood0, double ds) {
                 (void) ds;
                 return egret::Quadrupole::transfer_matrix(self.length, self.k1, self.tilt, cood0.delta);
             }, py::arg("cood0"), py::arg("ds") = 0.1)
        .def("transfer",
             [] (const QuadrupoleWrapper &self, const egret::Coordinate &cood0, double ds) {
                 (void) ds;
                 Eigen::Matrix4d t = egret::Quadrupole::transfer_matrix(self.length, self.k1, self.tilt, cood0.delta);
                 Eigen::Vector4d v = t * cood0.vector;
                 egret::Coordinate out(v, cood0.s + self.length, cood0.z, cood0.delta);
                 return py::make_tuple(out, py::none(), py::none());
             }, py::arg("cood0"), py::arg("ds") = 0.1)
        .def("transfer_matrix_array",
             [] (const QuadrupoleWrapper &self, const egret::Coordinate &cood0, double ds, bool endpoint) {
                 auto pr = egret::Quadrupole::transfer_matrix_array(self.length, self.k1, self.tilt, cood0.delta, ds, endpoint);
                 auto &tensor = pr.first;
                 auto &svec = pr.second;
                 int n = static_cast<int>(tensor.dimension(2));
                 py::array_t<double> arr({4,4,n});
                 auto buf = arr.mutable_unchecked<3>();
                 for (int k=0;k<n;++k) for (int i=0;i<4;++i) for (int j=0;j<4;++j) buf(i,j,k) = tensor(i,j,k);
                 py::array_t<double> sarr({n});
                 auto sbuf = sarr.mutable_unchecked<1>();
                 for (int i=0;i<n;++i) sbuf(i) = svec[i];
                 return py::make_tuple(arr, sarr);
             }, py::arg("cood0"), py::arg("ds") = 0.1, py::arg("endpoint") = true)
        .def("dispersion",
             [] (const QuadrupoleWrapper &self, const egret::Coordinate &cood0) {
                 return egret::Quadrupole::dispersion(cood0.vector, self.length, self.k1, cood0.delta);
             }, py::arg("cood0"))
        .def(py::pickle(
            [](const QuadrupoleWrapper &q) {
                return py::make_tuple(q.length, q.k1, q.tilt);
            },
            [](py::tuple t) {
                if (t.size() != 3) throw std::runtime_error("Invalid Quadrupole pickle");
                return std::make_shared<QuadrupoleWrapper>(t[0].cast<double>(), t[1].cast<double>(), t[2].cast<double>());
            }
        ));

    py::class_<DipoleWrapper, egret::Element, std::shared_ptr<DipoleWrapper>>(m, "Dipole", py::dynamic_attr())
        .def(py::init<double, double, double>(), py::arg("length"), py::arg("angle"), py::arg("k1") = 0.0)
        .def_readwrite("length", &DipoleWrapper::length)
        .def_readwrite("angle", &DipoleWrapper::angle)
        .def_readwrite("k1", &DipoleWrapper::k1)
        .def("transfer_matrix",
             [] (const DipoleWrapper &self, const egret::Coordinate &cood0, double ds) {
                 (void) ds;
                 return egret::Dipole::transfer_matrix(self.length, self.angle, self.k1, cood0.delta);
             }, py::arg("cood0"), py::arg("ds") = 0.1)
        .def("transfer",
             [] (const DipoleWrapper &self, const egret::Coordinate &cood0, double ds) {
                 (void) ds;
                 Eigen::Matrix4d t = egret::Dipole::transfer_matrix(self.length, self.angle, self.k1, cood0.delta);
                 Eigen::Vector4d v = t * cood0.vector;
                 egret::Coordinate out(v, cood0.s + self.length, cood0.z, cood0.delta);
                 return py::make_tuple(out, py::none(), py::none());
             }, py::arg("cood0"), py::arg("ds") = 0.1)
        .def("transfer_matrix_array",
             [] (const DipoleWrapper &self, const egret::Coordinate &cood0, double ds, bool endpoint) {
                 auto pr = egret::Dipole::transfer_matrix_array(self.length, self.angle, self.k1, cood0.delta, ds, endpoint);
                 auto &tensor = pr.first;
                 auto &svec = pr.second;
                 int n = static_cast<int>(tensor.dimension(2));
                 py::array_t<double> arr({4,4,n});
                 auto buf = arr.mutable_unchecked<3>();
                 for (int k=0;k<n;++k) for (int i=0;i<4;++i) for (int j=0;j<4;++j) buf(i,j,k) = tensor(i,j,k);
                 py::array_t<double> sarr({n});
                 auto sbuf = sarr.mutable_unchecked<1>();
                 for (int i=0;i<n;++i) sbuf(i) = svec[i];
                 return py::make_tuple(arr, sarr);
             }, py::arg("cood0"), py::arg("ds") = 0.1, py::arg("endpoint") = true)
        .def("dispersion",
             [] (const DipoleWrapper &self, const egret::Coordinate &cood0) {
                 // Some dipole implementations may not use cood0; delegate to static API
                 return egret::Dipole::transfer_matrix(self.length, self.angle, self.k1, cood0.delta), py::none();
             }, py::arg("cood0"))
        .def(py::pickle(
            [](const DipoleWrapper &d) {
                return py::make_tuple(d.length, d.angle, d.k1);
            },
            [](py::tuple t) {
                if (t.size() != 3) throw std::runtime_error("Invalid Dipole pickle");
                return std::make_shared<DipoleWrapper>(t[0].cast<double>(), t[1].cast<double>(), t[2].cast<double>());
            }
        ));

    py::class_<SextupoleWrapper, egret::Element, std::shared_ptr<SextupoleWrapper>>(m, "Sextupole", py::dynamic_attr())
        .def(py::init<double, double, double, double, double>(), py::arg("length"), py::arg("k2"), py::arg("dx") = 0.0, py::arg("dy") = 0.0, py::arg("ds") = 0.1)
        .def_readwrite("length", &SextupoleWrapper::length)
        .def_readwrite("k2", &SextupoleWrapper::k2)
        .def_readwrite("dx", &SextupoleWrapper::dx)
        .def_readwrite("dy", &SextupoleWrapper::dy)
        .def_readwrite("ds", &SextupoleWrapper::ds)
        .def("transfer_matrix_by_midpoint_method",
             [] (const SextupoleWrapper &self, const egret::Coordinate &cood0, double k0x, double k0y, double dx, double dy, double ds, bool tmatflag, bool dispflag) {
                 return egret::Sextupole::transfer_matrix_by_midpoint_method(cood0, self.length, self.k2, k0x, k0y, dx, dy, ds, tmatflag, dispflag);
             }, py::arg("cood0"), py::arg("k0x") = 0.0, py::arg("k0y") = 0.0, py::arg("dx") = 0.0, py::arg("dy") = 0.0, py::arg("ds") = 0.1, py::arg("tmatflag") = true, py::arg("dispflag") = false)
        .def("transfer_matrix",
             [] (const SextupoleWrapper &self, const egret::Coordinate &cood0, double ds) {
                 return egret::Sextupole::transfer_matrix(cood0, self.length, self.k2, ds);
             }, py::arg("cood0"), py::arg("ds") = 0.1)
        .def("transfer_matrix_array",
             [] (const SextupoleWrapper &self, const egret::Coordinate &cood0, double ds, bool endpoint) {
                 auto pr = egret::Sextupole::transfer_matrix_array(cood0, self.length, self.k2, ds, endpoint);
                 auto &tensor = pr.first;
                 auto &svec = pr.second;
                 int n = static_cast<int>(tensor.dimension(2));
                 py::array_t<double> arr({4,4,n});
                 auto buf = arr.mutable_unchecked<3>();
                 for (int k=0;k<n;++k) for (int i=0;i<4;++i) for (int j=0;j<4;++j) buf(i,j,k) = tensor(i,j,k);
                 return py::make_tuple(arr, svec);
             }, py::arg("cood0"), py::arg("ds") = 0.1, py::arg("endpoint") = true)
        .def("transfer",
             [] (const SextupoleWrapper &self, const egret::Coordinate &cood0, double ds) {
                 Eigen::Matrix4d t = egret::Sextupole::transfer_matrix(cood0, self.length, self.k2, ds);
                 Eigen::Vector4d v = t * cood0.vector;
                 egret::Coordinate out(v, cood0.s + self.length, cood0.z, cood0.delta);
                 return py::make_tuple(out, py::none(), py::none());
             }, py::arg("cood0"), py::arg("ds") = 0.1)
        .def("dispersion",
             [] (const SextupoleWrapper &self, const egret::Coordinate &cood0, double ds) {
                 return egret::Sextupole::dispersion(cood0, self.length, self.k2, ds);
             }, py::arg("cood0"), py::arg("ds") = 0.1)
        .def(py::pickle(
            [](const SextupoleWrapper &s) {
                return py::make_tuple(s.length, s.k2, s.dx, s.dy, s.ds);
            },
            [](py::tuple t) {
                if (t.size() != 5) throw std::runtime_error("Invalid Sextupole pickle");
                return std::make_shared<SextupoleWrapper>(t[0].cast<double>(), t[1].cast<double>(), t[2].cast<double>(), t[3].cast<double>(), t[4].cast<double>());
            }
        ));

    // Helper to call Element::transfer from Python to exercise C++ -> Python dispatch
    m.def("call_element_transfer", [] (std::shared_ptr<egret::Element> elem, const egret::Coordinate &cood0, double ds) {
        return elem->transfer(cood0, ds);
    });

    m.def("call_element_transfer_array", [] (std::shared_ptr<egret::Element> elem, const egret::Coordinate &cood0, double ds, bool endpoint) {
        return elem->transfer_array(cood0, ds, endpoint);
    });

        m.def("quadrupole_transfer_matrix", &egret::Quadrupole::transfer_matrix,
            "Compute 4x4 transfer matrix for a quadrupole",
            py::arg("length"), py::arg("k1"), py::arg("tilt") = 0.0, py::arg("delta") = 0.0);

        m.def("quadrupole_transfer_matrix_array",
            [] (double length, double k1, double tilt, double delta, double ds, bool endpoint) {
                auto pr = egret::Quadrupole::transfer_matrix_array(length, k1, tilt, delta, ds, endpoint);
                auto &tensor = pr.first;
                auto &svec = pr.second;
                int n = static_cast<int>(tensor.dimension(2));
                // create numpy array with shape (4,4,n)
                py::array_t<double> arr({4,4,n});
                auto buf = arr.mutable_unchecked<3>();
                for (int k=0;k<n;++k) for (int i=0;i<4;++i) for (int j=0;j<4;++j) buf(i,j,k) = tensor(i,j,k);
                py::array_t<double> sarr({n});
                auto sbuf = sarr.mutable_unchecked<1>();
                for (int i=0;i<n;++i) sbuf(i) = svec[i];
                return py::make_tuple(arr, sarr);
            },
            "Compute transfer matrix array for a quadrupole",
            py::arg("length"), py::arg("k1"), py::arg("tilt") = 0.0, py::arg("delta") = 0.0, py::arg("ds") = 0.1, py::arg("endpoint") = true);

        m.def("dipole_transfer_matrix", &egret::Dipole::transfer_matrix,
            "Compute 4x4 transfer matrix for a dipole",
            py::arg("length"), py::arg("angle"), py::arg("k1") = 0.0, py::arg("delta") = 0.0);

        m.def("dipole_transfer_matrix_array",
            [] (double length, double angle, double k1, double delta, double ds, bool endpoint) {
                auto pr = egret::Dipole::transfer_matrix_array(length, angle, k1, delta, ds, endpoint);
                auto &tensor = pr.first;
                auto &svec = pr.second;
                int n = static_cast<int>(tensor.dimension(2));
                py::array_t<double> arr({4,4,n});
                auto buf = arr.mutable_unchecked<3>();
                for (int k=0;k<n;++k) for (int i=0;i<4;++i) for (int j=0;j<4;++j) buf(i,j,k) = tensor(i,j,k);
                {
                    int ns = static_cast<int>(svec.size());
                    py::array_t<double> sarr({ns});
                    auto sbuf = sarr.mutable_unchecked<1>();
                    for (int i=0;i<ns;++i) sbuf(i) = svec[i];
                    return py::make_tuple(arr, sarr);
                }
            },
            "Compute transfer matrix array for a dipole",
            py::arg("length"), py::arg("angle"), py::arg("k1") = 0.0, py::arg("delta") = 0.0, py::arg("ds") = 0.1, py::arg("endpoint") = true);

            m.def("sextupole_transfer_matrix_by_midpoint_method", &egret::Sextupole::transfer_matrix_by_midpoint_method,
                "Single-step midpoint transfer for sextupole",
                py::arg("cood0"), py::arg("length"), py::arg("k2"), py::arg("k0x") = 0.0, py::arg("k0y") = 0.0,
                py::arg("dx") = 0.0, py::arg("dy") = 0.0, py::arg("ds") = 0.1, py::arg("tmatflag") = true, py::arg("dispflag") = false);

            m.def("sextupole_transfer_matrix", &egret::Sextupole::transfer_matrix,
                "Transfer matrix of sextupole (compose midpoint steps)",
                py::arg("cood0"), py::arg("length"), py::arg("k2"), py::arg("ds") = 0.1);

            m.def("sextupole_transfer_matrix_array",
                [] (const egret::Coordinate &cood0, double length, double k2, double ds, bool endpoint) {
                    auto pr = egret::Sextupole::transfer_matrix_array(cood0, length, k2, ds, endpoint);
                    auto &tensor = pr.first;
                    auto &svec = pr.second;
                    int n = static_cast<int>(tensor.dimension(2));
                    py::array_t<double> arr({4,4,n});
                    auto buf = arr.mutable_unchecked<3>();
                    for (int k=0;k<n;++k) for (int i=0;i<4;++i) for (int j=0;j<4;++j) buf(i,j,k) = tensor(i,j,k);
                    {
                        int ns = static_cast<int>(svec.size());
                        py::array_t<double> sarr({ns});
                        auto sbuf = sarr.mutable_unchecked<1>();
                        for (int i=0;i<ns;++i) sbuf(i) = svec[i];
                        return py::make_tuple(arr, sarr);
                    }
                },
                "Transfer matrix array of sextupole",
                py::arg("cood0"), py::arg("length"), py::arg("k2"), py::arg("ds") = 0.1, py::arg("endpoint") = true);

            m.def("sextupole_dispersion", &egret::Sextupole::dispersion,
                "Dispersion of sextupole",
                py::arg("cood0"), py::arg("length"), py::arg("k2"), py::arg("ds") = 0.1);

            // Apply a stack of 4x4 transfer matrices to one or many 4-vectors.
            // mats: ndarray shape (4,4,N)
            // vecs: ndarray shape (4,) or (4,M) or (M,4)
            // returns ndarray:
            //  - if vecs is 1D (4,), returns (4,N)
            //  - if vecs is 2D (4,M) or (M,4), returns (4,M,N)
            m.def("apply_transfer_matrix_array",
                [] (py::array mats, py::array vecs) -> py::object {
                    if (mats.ndim() != 3 || mats.shape(0) != 4 || mats.shape(1) != 4)
                        throw std::runtime_error("mats must have shape (4,4,N)");
                    int N = mats.shape(2);

                    // Ensure mats is in the expected C-contiguous layout (k fastest)
                    py::buffer_info mb = mats.request();
                    py::array mats_contig;
                    double *mp = nullptr;
                    bool mats_ok = false;
                    if (mb.ndim == 3) {
                        ptrdiff_t s2 = mb.strides[2] / sizeof(double);
                        ptrdiff_t s1 = mb.strides[1] / sizeof(double);
                        ptrdiff_t s0 = mb.strides[0] / sizeof(double);
                        if (s2 == 1 && s1 == static_cast<ptrdiff_t>(N) && s0 == static_cast<ptrdiff_t>(4) * s1) mats_ok = true;
                    }
                    if (!mats_ok) {
                        mats_contig = py::array_t<double, py::array::c_style | py::array::forcecast>(mats);
                        mb = mats_contig.request();
                    }
                    mp = static_cast<double*>(mb.ptr);

                    // vecs: support 1D length-4 (single vector) or total elements multiple of 4
                    py::buffer_info vb = vecs.request();
                    size_t nelems = 1;
                    for (int d = 0; d < vecs.ndim(); ++d) nelems *= static_cast<size_t>(vecs.shape(d));
                    if (nelems == 0) throw std::runtime_error("vecs must contain data");
                    if (nelems % 4 != 0) throw std::runtime_error("vecs must have total element count divisible by 4");
                    int M = static_cast<int>(nelems / 4);
                    bool single = (vecs.ndim() == 1 && nelems == 4);

                    // Setup vec access mode: 0=copy into vbuf, 1=strided direct access, 2=single-vector strides
                    int mode = 0;
                    double *vec_src = static_cast<double*>(vb.ptr);
                    ptrdiff_t s0 = 0, s1 = 0;
                    int dim0 = 0, dim1 = 0;
                    std::vector<double> vbuf;
                    if (single) {
                        mode = 2; // small single vector, read via strides when needed
                    } else {
                        if (vecs.ndim() == 1) {
                            // flat array of 4*M elements
                            if (vb.strides[0] / sizeof(double) == 1) {
                                mode = 1; // contiguous flat, linear indexing OK
                            } else {
                                // non-contiguous 1D: copy
                                mode = 0;
                            }
                        } else if (vecs.ndim() == 2) {
                            dim0 = vecs.shape(0); dim1 = vecs.shape(1);
                            s0 = vb.strides[0] / sizeof(double); s1 = vb.strides[1] / sizeof(double);
                            // Fast-path: contiguous C-layout with shape (4, M) -> strides (M,1)
                            if (dim0 == 4 && s1 == 1 && s0 == static_cast<ptrdiff_t>(dim1)) {
                                mode = 3; // fastest (4,M) layout
                            } else {
                                mode = 1; // allow general strided access
                            }
                        } else {
                            mode = 0;
                        }
                        if (mode == 0) {
                            // copy into contiguous 4 x M layout: vbuf[j + m*4]
                            vbuf.resize(4 * M);
                            double *src = vec_src;
                            if (vecs.ndim() == 1) {
                                for (size_t idx = 0; idx < nelems; ++idx) {
                                    int j = static_cast<int>(idx % 4);
                                    int m = static_cast<int>(idx / 4);
                                    vbuf[j + m*4] = src[idx * (vb.strides[0] / sizeof(double))];
                                }
                            } else {
                                // generic copy using strides
                                for (size_t idx = 0; idx < nelems; ++idx) {
                                    // map linear idx to multi-index and compute offset
                                    size_t tmp = idx;
                                    size_t off = 0;
                                    for (int d = vecs.ndim() - 1; d >= 0; --d) {
                                        int sd = vecs.shape(d);
                                        int coord = static_cast<int>(tmp % sd);
                                        off += coord * (vb.strides[d] / sizeof(double));
                                        tmp /= sd;
                                        if (d == 0) break;
                                    }
                                    int j = static_cast<int>(idx % 4);
                                    int m = static_cast<int>(idx / 4);
                                    vbuf[j + m*4] = src[off];
                                }
                            }
                        }
                    }

                    if (single) {
                        // Small-workload fallback: use NumPy for tiny work to avoid C++ overhead
                        const int SMALL_WORK_THRESHOLD = 1024;
                        int work = N * 1;
                        if (work < SMALL_WORK_THRESHOLD) {
                            py::module np = py::module::import("numpy");
                            // einsum 'ijk,j->ik' computes (4,N) directly
                            py::object res = np.attr("einsum")(py::str("ijk,j->ik"), mats, vecs);
                            return res;
                        }

                        // out shape (4,N)
                        py::array_t<double> out({4, N});
                        py::buffer_info ob = out.request();
                        double *op = static_cast<double*>(ob.ptr);
                        ptrdiff_t stride_elem = (vb.ndim == 1) ? (vb.strides[0] / sizeof(double)) : 0;

                        // Micro-optimization: cache single-vector components
                        double cv0, cv1, cv2, cv3;
                        if (mode == 2) {
                            cv0 = vec_src[0 * stride_elem];
                            cv1 = vec_src[1 * stride_elem];
                            cv2 = vec_src[2 * stride_elem];
                            cv3 = vec_src[3 * stride_elem];
                        } else if (mode == 1) {
                            cv0 = vec_src[0]; cv1 = vec_src[1]; cv2 = vec_src[2]; cv3 = vec_src[3];
                        } else {
                            cv0 = vbuf[0]; cv1 = vbuf[1]; cv2 = vbuf[2]; cv3 = vbuf[3];
                        }

#ifdef USE_OPENMP
                        #pragma omp parallel for schedule(static)
                        for (int k = 0; k < N; ++k) {
                            for (int i = 0; i < 4; ++i) {
                                int base = ((i*4)+0)*N + k;
                                double acc = mp[base + 0 * N] * cv0
                                           + mp[base + 1 * N] * cv1
                                           + mp[base + 2 * N] * cv2
                                           + mp[base + 3 * N] * cv3;
                                op[i * N + k] = acc;
                            }
                        }
#else
                        for (int k = 0; k < N; ++k) {
                            for (int i = 0; i < 4; ++i) {
                                int base = ((i*4)+0)*N + k;
                                double acc = mp[base + 0 * N] * cv0
                                           + mp[base + 1 * N] * cv1
                                           + mp[base + 2 * N] * cv2
                                           + mp[base + 3 * N] * cv3;
                                op[i * N + k] = acc;
                            }
                        }
#endif
                        return py::cast<py::object>(out);
                    } else {
                        // out shape (4,M,N)
                        py::array_t<double> out({4, M, N});
                        py::buffer_info ob = out.request();
                        double *op = static_cast<double*>(ob.ptr);
                        int work = N * std::max(1, M);
                        bool do_parallel = false;
#ifdef USE_OPENMP
                        const int PARALLEL_THRESHOLD = 4096;
                        do_parallel = (work >= PARALLEL_THRESHOLD);
#endif
                        if (do_parallel) {
#ifdef USE_OPENMP
                            #pragma omp parallel for collapse(2) schedule(static)
                            for (int k = 0; k < N; ++k) {
                                for (int m = 0; m < M; ++m) {
                                    for (int i = 0; i < 4; ++i) {
                                        int base = ((i*4)+0)*N + k;
                                        double v0, v1, v2, v3;
                                        if (mode == 3) {
                                            v0 = vec_src[0 * dim1 + m];
                                            v1 = vec_src[1 * dim1 + m];
                                            v2 = vec_src[2 * dim1 + m];
                                            v3 = vec_src[3 * dim1 + m];
                                        } else if (mode == 1) {
                                            if (dim0 == 4) {
                                                v0 = vec_src[0 * s0 + m * s1];
                                                v1 = vec_src[1 * s0 + m * s1];
                                                v2 = vec_src[2 * s0 + m * s1];
                                                v3 = vec_src[3 * s0 + m * s1];
                                            } else if (dim1 == 4) {
                                                v0 = vec_src[m * s0 + 0 * s1];
                                                v1 = vec_src[m * s0 + 1 * s1];
                                                v2 = vec_src[m * s0 + 2 * s1];
                                                v3 = vec_src[m * s0 + 3 * s1];
                                            } else {
                                                size_t base_idx = static_cast<size_t>(m) * 4;
                                                v0 = vec_src[base_idx + 0]; v1 = vec_src[base_idx + 1]; v2 = vec_src[base_idx + 2]; v3 = vec_src[base_idx + 3];
                                            }
                                        } else if (mode == 2) {
                                            // single-vector-like stride access (shouldn't reach here for multi)
                                            v0 = vec_src[0 * (vb.strides[0] / sizeof(double))];
                                            v1 = vec_src[1 * (vb.strides[0] / sizeof(double))];
                                            v2 = vec_src[2 * (vb.strides[0] / sizeof(double))];
                                            v3 = vec_src[3 * (vb.strides[0] / sizeof(double))];
                                        } else {
                                            v0 = vbuf[0 + m*4]; v1 = vbuf[1 + m*4]; v2 = vbuf[2 + m*4]; v3 = vbuf[3 + m*4];
                                        }
                                        double acc = mp[base + 0 * N] * v0
                                                   + mp[base + 1 * N] * v1
                                                   + mp[base + 2 * N] * v2
                                                   + mp[base + 3 * N] * v3;
                                        op[(i * M + m) * N + k] = acc;
                                    }
                                }
                            }
#endif
                        } else {
                            for (int k = 0; k < N; ++k) {
                                for (int m = 0; m < M; ++m) {
                                    for (int i = 0; i < 4; ++i) {
                                        int base = ((i*4)+0)*N + k;
                                        double v0, v1, v2, v3;
                                        if (mode == 3) {
                                            v0 = vec_src[0 * dim1 + m];
                                            v1 = vec_src[1 * dim1 + m];
                                            v2 = vec_src[2 * dim1 + m];
                                            v3 = vec_src[3 * dim1 + m];
                                        } else if (mode == 1) {
                                            if (dim0 == 4) {
                                                v0 = vec_src[0 * s0 + m * s1];
                                                v1 = vec_src[1 * s0 + m * s1];
                                                v2 = vec_src[2 * s0 + m * s1];
                                                v3 = vec_src[3 * s0 + m * s1];
                                            } else if (dim1 == 4) {
                                                v0 = vec_src[m * s0 + 0 * s1];
                                                v1 = vec_src[m * s0 + 1 * s1];
                                                v2 = vec_src[m * s0 + 2 * s1];
                                                v3 = vec_src[m * s0 + 3 * s1];
                                            } else {
                                                size_t base_idx = static_cast<size_t>(m) * 4;
                                                v0 = vec_src[base_idx + 0]; v1 = vec_src[base_idx + 1]; v2 = vec_src[base_idx + 2]; v3 = vec_src[base_idx + 3];
                                            }
                                        } else if (mode == 2) {
                                            v0 = vec_src[0 * (vb.strides[0] / sizeof(double))];
                                            v1 = vec_src[1 * (vb.strides[0] / sizeof(double))];
                                            v2 = vec_src[2 * (vb.strides[0] / sizeof(double))];
                                            v3 = vec_src[3 * (vb.strides[0] / sizeof(double))];
                                        } else {
                                            v0 = vbuf[0 + m*4]; v1 = vbuf[1 + m*4]; v2 = vbuf[2 + m*4]; v3 = vbuf[3 + m*4];
                                        }
                                        double acc = mp[base + 0 * N] * v0
                                                   + mp[base + 1 * N] * v1
                                                   + mp[base + 2 * N] * v2
                                                   + mp[base + 3 * N] * v3;
                                        op[(i * M + m) * N + k] = acc;
                                    }
                                }
                            }
                        }
                        return py::cast<py::object>(out);
                    }
                },
                "Apply stack of 4x4 transfer matrices to vectors",
                py::arg("mats"), py::arg("vecs"));
#endif
}
