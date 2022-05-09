#include <pybind11/pybind11.h>

#include "decoder/ctc_prefix_beam_search.h"

namespace py = pybind11;

// data shape: [max_time, bs, num_classes]
// decoded shape: [bs, top_paths, max_time] ignore id: -1
// log_probs shape: [bs, top_paths] ignore id: -1
void ctc_beam_search_decoder(float *data, int max_time, int bs, int num_classes,
                             int *sequence_length, int n_sequence_length,
                             int beam_width, int top_path, int *decoded,
                             float *log_probs) {
  if (data == nullptr) {
    return;
  }
  if (max_time <= 0 || bs <= 0 || num_classes <= 0 || n_sequence_length <= 0 ||
      beam_width <= 0 || top_path <= 0) {
    return;
  }

  wenet::CtcPrefixBeamSearchOptions opts{beam_width, top_path};
  wenet::SearchInterface searcher;
  searcher.reset(new wenet::CtcPrefixBeamSearch(opts));

  for (int i = 0; i < bs; i++) {
    std::vector<std::vector<float>> ctc_log_probs;
    // fill ctc_log_probs
    // copy data[i,:sequence_length[i], :] to ctc_log_probs
    for (int j = 0; j < sequence_length[i]; j++) {
      std::vector<float> rows(num_classes, 0.0);
      std::copy(data + i * max_time * num_classes,
                data + i * (max_time * num_classes) + j * num_classes,
                rows.begin());

      ctc_log_probs.emplace_back(std::move(rows));
    }
    searcher_->Search(ctc_log_probs);
    const auto &hypotheses = searcher_->Inputs();
    const auto &likelihood = searcher_->Likelihood();
    // TODO: support export timestamp
    // fill timestamp
    // const auto &times = searcher_->Times();

    // fill decoded
    // fill log_probs
  }
}

PYBIND11_MODULE(_ctc_decoder, m) {
  m.doc() = "ctc decoder"; // optional module docstring
}
