# ctcdecoder
ctc decoder binding for wenet runtime

TODO:
- [ ] suport batch ctc decode
- [ ] suport chunk state ctc decode
- [ ] suport torch sparse tensor 

```python
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
print(decoder.decode(inputs, seq_len))
[[tensor([2, 1]), tensor([1, 2]), tensor([1])], 
[tensor([2, 1]), tensor([1, 2]), tensor([1])]]
```


