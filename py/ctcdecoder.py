from typing import List, Tuple

import _ctcdecoder
import torch


class CTCDecoder:
    def __init__(self, beam_width: int =10, top_paths: int =10):
        self.beam_width = beam_width
        self.top_paths = top_paths

    def decode(self,
               inputs: torch.Tensor,
               sequence_length: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """ Decode the input data
        Args:
            inputs: shape [batch_size, max_step, num_classes], data_type torch.float32
            sequence_length: shape [batch_size]
        Returns:
           nbest_decoded: shape [batch_size, top_path, max_seq_len]
           log_probability: shape [batch_size, top_paths]
        """
        # TODO: assert inputs shape
        max_time = inputs.size(1)
        bs = inputs.size(0)
        num_classes = inputs.size(2)
        # n_sequence = sequence_length.size(0)
        decoder_result = _ctcdecoder.ctc_beam_search_decoder(
            inputs.data_ptr(),
            max_time,
            bs,
            num_classes,
            sequence_length.data_ptr(),
            num_classes,
            self.beam_width,
            self.top_paths,
        )
        # TODO: timestamp and prob
        # b_nbest = []
        # log_probs = []
        #  [bs, top_path, max_seq_len]
        # TODO: fill torch.tensor in c++ not here

        n_nbest = []
        n_nlog_probs = []
        for nth in decoder_result:
            nbest = []
            for onebest in nth.hypotheses:
                nbest.append(torch.tensor(onebest))

            n_nbest.append(torch.nn.utils.rnn.pad_sequence(nbest, batch_first=True, padding_value=-1))

            n_nlog_probs.append(torch.tensor(nth.likelihood))

        n_nlog_probs = torch.nn.utils.rnn.pad_sequence(n_nlog_probs, batch_first=True, padding_value=-1)

        return n_nbest, n_nlog_probs

