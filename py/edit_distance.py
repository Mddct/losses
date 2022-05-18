import _ctcdecoder
import torch


# std::vector<int64_t> EditDistance(uintptr_t sdata, int s_max_seq_len,
#                                   uintptr_t s_sequence_length, uintptr_t tdata,
#                                   int t_max_seq_len,
#                                   uintptr_t t_sequence_length, int bs);
#
def edit_distance(hypothesis: torch.Tensor,
                  h_seq_lens: torch.Tensor,
                  truth: torch.Tensor,
                  t_seq_lens: torch.Tensor)-> torch.Tensor:
    # TODO: check tensor shape
    assert hypothesis.size(0) == truth.size(0)

    bs = hypothesis.size(0)
    hyp_max_seq_len = hypothesis.size(1)
    truth_max_seq_len = truth.size(1)

    assert h_seq_lens.size(0) == bs
    assert t_seq_lens.size(0) == bs

    distances = _ctcdecoder.edit_distance(
        hypothesis.data_ptr(),
        hyp_max_seq_len,
        h_seq_lens.data_ptr(),
        truth.data_ptr(),
        truth_max_seq_len,
        t_seq_lens.data_ptr(),
        bs,
    )
    return torch.tensor(distances, dtype=torch.int64)
