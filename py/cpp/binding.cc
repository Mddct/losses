#include "ctc_decoder.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_ctcdecoder, m) {
  m.doc() = "ctc decoder"; // optional module docstring

  py::class_<DecodeResult>(m, "DecodeResult")
      .def(py::init<>())
      .def_readwrite("hypotheses", &DecodeResult::hypotheses)
      .def_readwrite("likelihood", &DecodeResult::likelihood);

  m.def("ctc_beam_search_decoder", &ctc_beam_search_decoder,
        py::return_value_policy::reference, "ctc prefix beam searcher decode");
  m.def("edit_distance", &EditDistance, py::return_value_policy::reference,
        "edit distance");
}
