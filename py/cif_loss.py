import torch

class CtcBoundaryLoss(torch.nn.Module):
    """ https://arxiv.org/pdf/2104.04702.pdf
    """

    def __init__(self,
                 spike_threshold: float = 0.0,
                 blank_id: int = 0) -> None:
        super().__init__()
        self.spike_threshold = math.log(spike_threshold)
        self.blank = blank_id

    def forward(self, alpha: torch.Tensor, ctc_log_probs: torch.Tensor,
                mask: torch.Tensor):

        batch_size = alpha.size(0)
        ctc_blank_probs = ctc_log_probs[:, :, self.blank]
        triggerd = (1 - ctc_blank_probs) > self.spike_threshold
        spikes = triggerd * mask
        begin = torch.ones(batch_size,
                           1,
                           dtype=spikes.dtype,
                           device=spikes.device)
        spikes = torch.cat([begin, spikes], dim=1)
        index = torch.arange(alpha.size(1),
                             device=alpha.device).unsqueeze(0)  #[1,L]
        boundary_loss_list = []

        # TODO(Mddct): refactor later by vector
        for (i, spike) in enumerate(spikes):
            spike = torch.nonzero(spike).squeeze(1)
            if torch.sum(spike) > 0:
                # [] or [1] not considered
                start = spike[:-1]
                end = spike[1:]
                m = index >= start.unsqueeze(1)
                m = m <= end.unsqueeze(1)

                loss_i_j = alpha[i:i + 1, :] * m  # [1, L]
                loss_i_j = torch.sum(loss_i_j, 1)
                boundary_loss_list.append(torch.sum(torch.abs(loss_i_j - 1)))
        if len(boundary_loss_list) > 0:
            loss = torch.stack(boundary_loss_list, dim=0)
            return loss.sum() / loss.size(0)
        else:
            return torch.tensor(0.0, dtype=torch.float32,
                                device=alpha.device).detach()
