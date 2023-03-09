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
        # alpha: torch.Tensor [B, T]
        # boundary: torch.Tensor [B,T]
        # mask: [B,T]
        batch_size = alpha.size(0)
        with torch.no_grad():
            top_1_prob, top_1_index = torch.topk(ctc_log_probs, 1,
                                                 dim=2)  # [B,T,1]
            first_dumy_prob = torch.tensor(
                [self.spike_threshold + 0.1] * batch_size,
                dtype=ctc_log_probs.dtype,
                device=ctc_log_probs.device).unsqueeze(1)
            top_1_prob = top_1_prob.squeeze(2)
            top_1_index = top_1_index.squeeze(2)
            first_dumy_index = torch.tensor(
                [1] * batch_size,
                dtype=first_dumy_prob.dtype,
                device=first_dumy_prob.device).unsqueeze(1)
            top_1_prob = torch.cat([first_dumy_prob, top_1_prob[:, 1:]], dim=1)
            top_1_index = torch.cat([first_dumy_index, top_1_index[:, 1:]],
                                    dim=1)

            spike = torch.greater(top_1_prob, self.spike_threshold)
            non_blank_mask = (top_1_index != self.blank)
            spikes = spike * mask * non_blank_mask

        index = torch.arange(alpha.size(1),
                             device=alpha.device).unsqueeze(0)  #[1,L]
        boundary_loss_list = []

        # TODO(Mddct): refactor later by vector
        for (i, spike) in enumerate(spikes):
            spike = torch.nonzero(spike)
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
