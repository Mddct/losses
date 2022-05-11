from torch.nn.utils.rnn import pad_sequence
from ctcdecoder import CTCDecoder
import torch

inputs = torch.tensor(
        [[[0.25, 0.40, 0.35],
           [0.40, 0.35, 0.25],
           [0.10, 0.50, 0.40]]]);
inputs = inputs.log()
seq_len = torch.tensor([3])
decoder = CTCDecoder(3,3)
print(pad_sequence(decoder.decode(inputs, seq_len), batch_first=True, padding_value=-1))

#tensor([[ 2,  1],
#        [ 1,  2],
#        [ 1, -1]])
#
