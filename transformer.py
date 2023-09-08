import util
import torch
from torch import nn
from encoder import RTIDS_Encoder
from decoder import RTIDS_Decoder

class RTIDS_Transformer(nn.Module):
    def __init__(self, trg_vocab, d_model, N, heads, dropout = 0.1):
        super().__init__()
        self.encoder = RTIDS_Encoder(d_model, N , heads, dropout)
        self.decoder = RTIDS_Decoder(d_model, N , heads, dropout)
        self.out = nn.Linear(78*d_model, trg_vocab)
    
    def forward(self, src, trg_mask=None):
        e_outputs = self.encoder(src, None)
        d_output = self.decoder(src, e_outputs, trg_mask)
        d_intermediate = d_output.view(d_output.size(0), -1)
        output = self.out(d_intermediate)
        output = torch.softmax(output,dim=1)
        return output
    
class IDS_Encoder_Only(nn.Module):
    def __init__(self, trg_vocab, d_model, N, heads, dropout = 0.1):
        super().__init__()
        self.encoder = RTIDS_Encoder(d_model, N, heads, dropout)
        self.out = nn.Linear(78*d_model, trg_vocab)

    def forward(self, src):
        e_outputs = self.encoder(src, None)
        e_intermediate = e_outputs.view(e_outputs.size(0),-1)
        output = self.out(e_intermediate)
        return torch.softmax(output, dim=1)
