#include "decoder/ctc_prefix_beam_search.h"
#include "decoder/search_interface.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

struct DecodeResult {
  std::vector<std::vector<int>> hypotheses;
  std::vector<float> likelihood;
  
  explicit DecodeResult() {}
};

namespace py = pybind11;

// data shape: [bs, max_time, num_classes]
// decoded shape: [bs, top_paths, max_time] ignore id: -1
// log_probs shape: [bs, top_paths] ignore id: -1
std::vector<DecodeResult>
ctc_beam_search_decoder(uintptr_t pdata, int max_time, int bs, int num_classes,
                        uintptr_t psequence_length, int n_sequence_length,
                        int beam_width, int top_path) {
  std::vector<DecodeResult> results;
  float *data = reinterpret_cast<float *>(pdata);
  if (max_time <= 0 || bs <= 0 || num_classes <= 0 || n_sequence_length <= 0 ||
      beam_width <= 0 || top_path <= 0) {
    return results;
  }
  int *sequence_length = reinterpret_cast<int *>(psequence_length);

  wenet::CtcPrefixBeamSearchOptions opts;
  opts.first_beam_size = beam_width;
  opts.second_beam_size = top_path;
  std::unique_ptr<wenet::SearchInterface> searcher;
  searcher.reset(new wenet::CtcPrefixBeamSearch(opts));

  for (int i = 0; i < bs; i++) {
    searcher->Reset();
    std::vector<std::vector<float>> ctc_log_probs;
    // fill ctc_log_probs
    // copy data[i,:sequence_length[i], :] to ctc_log_probs
    for (int j = 0; j < sequence_length[i]; j++) {
      std::vector<float> rows(num_classes, 0.0);
      std::copy(data + i * max_time * num_classes + j * num_classes,
                data + i * (max_time * num_classes) + (j+1)*num_classes,
                rows.begin());

      ctc_log_probs.emplace_back(std::move(rows));
    }

    searcher->Search(ctc_log_probs);
    searcher->FinalizeSearch();
    const auto &hypotheses = searcher->Inputs();
    const auto &likelihood = searcher->Likelihood();

    // TODO: support export timestamp
    // fill timestamp
    // const auto &times = searcher_->Times();
    
    DecodeResult result;
    for (auto i = 0; i < hypotheses.size(); i++) {
      result.hypotheses.push_back(std::move(hypotheses[i]));
    }
    result.likelihood.resize(likelihood.size());
    std::copy(likelihood.begin(), likelihood.end(), result.likelihood.begin());
    results.emplace_back(std::move(result));
  }
  return std::move(results);
}

PYBIND11_MODULE(_ctcdecoder, m) {
  m.doc() = "ctc decoder"; // optional module docstring

  py::class_<DecodeResult>(m, "DecodeResult")
      .def(py::init<>())
      .def_readwrite("hypotheses", &DecodeResult::hypotheses)
      .def_readwrite("likelihood", &DecodeResult::likelihood);

  m.def("ctc_beam_search_decoder", &ctc_beam_search_decoder,
        py::return_value_policy::reference, "ctc prefix beam searcher decode");
}

