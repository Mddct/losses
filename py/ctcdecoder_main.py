import torch
from torch.nn.utils.rnn import pad_sequence

from ctcdecoder import CTCDecoder
from edit_distance import edit_distance

inputs = torch.tensor(
        [[[0.25, 0.40, 0.35],
           [0.40, 0.35, 0.25],
           [0.10, 0.50, 0.40]]]);
inputs = inputs.log()
seq_len = torch.tensor([3])
decoder = CTCDecoder(3,3)
print(decoder.decode(inputs, seq_len))
# print(pad_sequence(decoder.decode(inputs, seq_len), batch_first=True, padding_value=-1))

#tensor([[ 2,  1],
#        [ 1,  2],
#        [ 1, -1]])
#

hyp = torch.tensor([[1,2,3], [1,2,3]])
hyp_lens = torch.tensor([3,3])
truth = torch.tensor([[4,5,6], [4, 5, 6]])
t_lens = torch.tensor([3,3])

print(edit_distance(hyp,hyp_lens,truth, t_lens)
