from typing import List

import _ctcdecoder
import torch


class CTCDecoder:
    def __init__(self, beam_width: int =10, top_paths: int =10):
        self.beam_width = beam_width
        self.top_paths = top_paths

    def decode(self, inputs: torch.Tensor, sequence_length: torch.Tensor) -> List[List[torch.Tensor]]:
        """ Decode the input data
        Args:
            inputs: shape [batch_size, max_step, num_classes], data_type torch.float32
            sequence_length: shape [batch_size]
        """
        # TODO: assert inputs shape
        max_time = inputs.size(1)
        bs = inputs.size(0)
        num_classes = inputs.size(2)
        n_sequence = sequence_length.size(0)
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
        b_nbest = [ torch.tensor(result) for bs in decoder_result for result in bs.hypotheses]

        return b_nbest

