import torch

from ctcdecoder import CTCDecoder

class PFRLoss(torch.nn.Module):

    def __init__(self, tempature: float = 10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tempature = tempature

    def forward(self, logits: torch.Tensor, mask: torch.Tensor):
        """ logits: [B,T,V], log probs
            mask: [B,1,T]
        """
        # log softmax
        logits = logits.exp() / self.tempature
        stu = logits[:, 1:, :]
        tea = logits[:, :-1, :]

        stu = torch.nn.functional.log_softmax(stu)
        tea = torch.nn.functional.softmax(tea)

        kl = torch.nn.functional.kl_div(stu, tea, reduction='none')
        kl = kl * mask[:, 1:, :]
        return kl.sum(-1).sum(-1) / kl.size()

class CTCKDLoss(torch.nn.Module):
    ''' CTCKDLoss is class for nbest strategy knowledge distill
    '''
    def __init__(
        self,
        nbest: int = 1,
    ):
        u""" [1]  https://arxiv.org/abs/2005.09310
        """ 
        super().__init__()
        self.ctc_prefix_beam_decoer = CTCDecoder(beam_width=nbest,top_paths=nbest)
        self.nbest = nbest
        self.ctc_loss = torch.nn.CTCLoss(reduction='sum')
        
    def forward(self, 
                s_logits: torch.Tensor,
                s_logits_length: torch.Tensor,
                t_logits: torch.Tensor,
                t_logits_length: torch.Tensor,      
               ) -> torch.Tensor:
        """Calculate focal loss.
        Args:
            s_logits: [bs, ...., n_class] student log probs
            s_logits_length: [bs]
            t_logits: [bs, ...., n_class] teacher log probos
            t_logits_length [bs]:
        Returns:
        """
        t_nbest_decoded, _ = self.ctc_prefix_beam_decoer.decode(
            t_logits, t_logits_length)
        # [batch_size, top_path, max_seq_len] -> [top_path, batch_size, max_seq_len]
        t_nbest = t_nbest.transpose(0,1)
        # (B, L, Nvocab) -> (L, B, Nvocab)
        s_logits = s_logits.transpose(0, 1)
        for (i,n) in enumerate(t_nbest_decoded):
          # n:  batch_size, max_seq_len]
          #s_onebest_hyp_lens: 
          s_onebest_hyp_lens = torch.sum(torch.where(n==-1, 0, 1), dim=1)
          
          loss.append(self.ctc_loss(s_logits, n, s_logits_length, s_onebest_hyp_lens)/s_logits_length.size(0))
          
        loss = torch.cat(loss, dim=0)
        return torch.sum(loss) / self.nbest
          
