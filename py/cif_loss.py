import torch

class CtcBoundaryLossV3(torch.nn.Module):
    """ https://arxiv.org/pdf/2104.04702.pdf
    """

    def __init__(self,
                 spike_threshold: float = 0.0,
                 blank_id: int = 0) -> None:
        super().__init__()
        self.spike_threshold = math.log(1-spike_threshold)
        self.blank = blank_id

    def forward(self, alpha: torch.Tensor, ctc_log_probs: torch.Tensor,
                mask: torch.Tensor, text_length: torch.Tensor):
        # alpha: torch.Tensor [B, T]
        # boundary: torch.Tensor [B,T]
        # mask: [B,T]
        text_mask = make_non_pad_mask(text_length)
        batch_size = alpha.size(0)
        ctc_blank_probs = ctc_log_probs[:, :, self.blank]
        triggerd = ctc_blank_probs < self.spike_threshold
        spikes = triggerd * mask
        index = torch.arange(alpha.size(1),
                             device=alpha.device).unsqueeze(0)  #[1,L]
        boundary_loss_list = []
        # math.log(1) == 0
        ones = torch.tensor([0.0],
                            dtype=alpha.dtype,
                            device=alpha.device,
                            requires_grad=False)
        # TODO(Mddct): refactor later by vector
        for (i, spike) in enumerate(spikes):
            spike = torch.nonzero(spike).squeeze(1)
            if torch.sum(spike) > 0:
                # [] or [1] not considered
                start = spike[:-1]
                end = spike[1:]
                l = index >= start.unsqueeze(1)
                r = index <= end.unsqueeze(1)
                m = l & r
                loss_i_j = alpha[i:i + 1, :] * m  # [1, L]
                loss_i_j = torch.sum(loss_i_j, 1)  # [L]
                boundary_loss_list.append(loss_i_j)
            else:
                boundary_loss_list.append(ones)
        boundary = torch.nn.utils.rnn.pad_sequence(boundary_loss_list,
                                                   batch_first=True)
        length = min(boundary.size(1), text_mask.size(1))
        mask = text_mask[:, :length]
        boundary = boundary[:, :length]

        return torch.sum(torch.abs(boundary - 1) * mask,
                         dim=1).sum() / batch_size
