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
            logits: [..., logit], positive log probos
        Returns:
            loss[i..., j] = - αₜ(1-pₜ)ˠlog(pₜ)
        """
        probs = torch.exp(logits)
        coefficient = torch.pow(1.0 - probs, self.gamma)
        if self.stop_gradient_on_focal_loss_coefficient:
          coefficient = coefficient.detach()
       
        return  -self.alpha * coefficient * logits

class SoftmaxCrossEntropyFocalLoss(torch.nn.Module):

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.5,
        stop_gradient_on_focal_loss_coefficient: bool = false,
    ):
          
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.stop_gradient_on_focal_loss_coefficient = stop_gradient_on_focal_loss_coefficient
        self.softmax_cross_entropy_loss = torch.nn.NLLLoss(reduction="none")

    def forward(self, 
                logits: torch.Tensor, 
                labels: torch.Tenor,
                labels_probs, torch.Tensor = None,
               ) -> torch.Tensor:
        """Calculate focal loss.
        (Note): using 
        Args:
            logits: logits before softmax because using nn.NLLLoss for numerically stable
        Returns:
            loss[i..., j] = - αₜ(1-pₜ)ˠlog(pₜ)
        """
        if labels_probs is not None:
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # average 
            loss = -(log_probs*logits)
        else:
            loss = self.softmax_cross_entropy_loss(logits, labels)
        
        # TODO: apply focal loss helper to resue code
        probs = torch.exp(-loss)
        coefficient = torch.pow(1.0 - probs, self.gamma)
        if self.stop_gradient_on_focal_loss_coefficient:
          coefficient = coefficient.detach()
       
        return  self.alpha * coefficient * loss

