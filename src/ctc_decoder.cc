#include "ctc_decoder.h"
#include <vector>

// data shape: [bs, max_time, num_classes]
// decoded shape: [bs, top_paths, max_time] ignore id: -1
// log_probs shape: [bs, top_paths] ignore id: -1
std::vector<DecodeResult>
ctc_beam_search_decoder(uintptr_t pdata, int max_time, int bs, int num_classes,
                        uintptr_t psequence_length, int n_sequence_length,
                        int beam_width, int top_path) {
  std::vector<DecodeResult> results;
  if (max_time <= 0 || bs <= 0 || num_classes <= 0 || n_sequence_length <= 0 ||
      beam_width <= 0 || top_path <= 0) {
    return results;
  }

  auto *data = reinterpret_cast<float *>(pdata);
  auto *sequence_length = reinterpret_cast<std::int64_t *>(psequence_length);

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
                data + i * (max_time * num_classes) + (j + 1) * num_classes,
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

std::vector<DecodeResult>
parallel_ctc_beam_search_decoder(uintptr_t pdata, int max_time, int bs,
                                 int num_classes, uintptr_t psequence_length,
                                 int n_sequence_length, int beam_width,
                                 int top_path) {
  if (max_time <= 0 || bs <= 0 || num_classes <= 0 || n_sequence_length <= 0 ||
      beam_width <= 0 || top_path <= 0) {
    return results;
  }

  auto *data = reinterpret_cast<float *>(pdata);
  auto *sequence_length = reinterpret_cast<std::int64_t *>(psequence_length);

  {
    // TODO: test for now change in future
    auto thread_num = bs;
    wenet::ThreadPool pool(thread_num);
    std::vector<std::future<std::vector<std::vector<DecodeResult>>>> res;
    for (int i = 0; i < bs; i++) {
      auto next =
          reinterpret_cast<uintptr_t>(data + i * max_time * num_classes);
      auto next_sequence_length =
          reinterpret_cast<uintptr_t>(sequence_length + i);

      res.emplace_back(pool.enqueue(ctc_beam_search_decoder, next, max_time, 1,
                                    num_classes, next_sequence_length, 1,
                                    beam_width, top_path));
    }

    // future result
    std::vector<DecodeResult> batch_results;
    for (int i =0 ; i < bs; i++){
      batch_results.emplace_back(res[i].get(0)[0]);
    }
  }
}
