#ifndef CTC_DECODER_H_
#define CTC_DECODER_H_

#include <cinttypes>
#include <vector>

struct DecodeResult {
  std::vector<std::vector<int>> hypotheses;
  std::vector<float> likelihood;

  explicit DecodeResult() {}
};

std::vector<DecodeResult> ctc_beam_search_decoder(uintptr_t pdata, int max_time,
                                                  int bs, int num_classes,
                                                  uintptr_t psequence_length,
                                                  int n_sequence_length,
                                                  int beam_width, int top_path);

std::vector<DecodeResult>
parallel_ctc_beam_search_decoder(uintptr_t pdata, int max_time, int bs,
                                 int num_classes, uintptr_t psequence_length,
                                 int n_sequence_length, int beam_width,
                                 int top_path);

std::vector<int64_t> EditDistance(uintptr_t sdata, int s_max_seq_len,
                                  uintptr_t s_sequence_length, uintptr_t tdata,
                                  int t_max_seq_len,
                                  uintptr_t t_sequence_length, int bs);

#endif // CTC_DECODER_H_
