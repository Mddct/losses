import torch

class SequenceFocalLoss(torch.nn.Module):
    ''' (Note): This is the simple version of the implementation。 Not for class based just for sequence leval
    '''
    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.5,
        stop_gradient_on_focal_loss_coefficient: bool = false,
    ):
        u""" [1] Focal loss https://arxiv.org/abs/1708.02002
        Args:
          stop_gradient_on_focal_loss_coefficient: If true, stops gradient on the
          focal loss coefficient (1-p)^gamma to stabilize the gradient.
        """
       
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.stop_gradient_on_focal_loss_coefficient = stop_gradient_on_focal_loss_coefficient

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss.
        Args:
            logits: [..., logit]
        Returns:
            loss[i..., j] = - αₜ(1-pₜ)ˠlog(pₜ)
        """
        probs = torch.exp(logits)
        coefficient = torch.pow(1.0 - probs, self.gamma)
        if self.stop_gradient_on_focal_loss_coefficient:
          coefficient = coefficient.detach()
       
        return  -self.alpha * coefficient * logits 

  # TODO: class based focal losses
