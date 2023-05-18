import util
import torch
from torch import nn
from encoder import RTIDS_Encoder
from decoder import RTIDS_Decoder

class RTIDS_Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout = 0.1):
        super().__init__()
        self.encoder = RTIDS_Encoder(src_vocab, d_model, N , heads, dropout)
        self.decoder = RTIDS_Decoder(trg_vocab, d_model, N , heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    
    def forward(self, src, trg, trg_mask):
        e_outputs = self.encoder(src, None)
        d_output = self.decoder(trg, e_outputs, trg_mask)
        output = self.out(d_output)
        return output