#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "stockfish.h"

namespace py = pybind11;

PYBIND11_MODULE(stockfish_module, m) {

    // data structures
    py::class_<StockfishMove>(m, "StockfishMove")
        .def(py::init<>())
        .def_readonly("move_letters", &StockfishMove::move_letters)
        .def_readonly("depth_evaluated", &StockfishMove::depth_evaluated)
        .def_readonly("move_eval", &StockfishMove::move_eval)
        .def_readonly("move_placement", &StockfishMove::move_placement)
        ;

    py::class_<StockfishWrapper>(m, "StockfishWrapper")
        .def(py::init<>())
        .def("init", &StockfishWrapper::init)
        .def("send_command", &StockfishWrapper::send_command)
        .def("generate_moves", &StockfishWrapper::generate_moves)
        .def_readwrite("target_depth", &StockfishWrapper::target_depth)
        .def_readwrite("num_threads", &StockfishWrapper::num_threads)
        .def_readwrite("num_lines", &StockfishWrapper::num_lines)
        .def_readwrite("elo_value", &StockfishWrapper::elo_value)
        ;

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}