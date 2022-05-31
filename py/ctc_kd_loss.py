import torch

from ctcdecoder import CTCDecoder

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
        loss = []
        ## wrong implement to fix it
        ## refine ctc decoder return nbest_decodes[j] is a batch one bestresult
        for (i,n) in enumerate(t_nbest_decoded):
          hyp_lens = torch.sum(torch.where(n==-1, 0, 1), dim=1)
          labels_repeat = labels[i].unsqueeze(0).repeat(n.size(0), 1)
          labels_length_repeat = labels_length[i].unsqueeze(0).repeat(n.size(0))
          
          # (B, L, Nvocab) -> (L, B, Nvocab)
          s_logits = s_logits.transpose(0, 1)
          loss.append(self.ctc_loss(s_logits, s_logits_length, s_logits_length, labels_length_repeat)/s_logits_length.size(0))
          
        loss = torch.cat(loss, dim=0)
        return torch.sum(loss) / self.nbest
          
