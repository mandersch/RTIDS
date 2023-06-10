import torch
from torch import nn
from util.embedder import RTIDS_Embedder
from util.positional_encoder import RTIDS_Positional_Encoder
from util.norm import RTIDS_Norm
from util.feed_forward import RTIDS_FeedForward
from util.multi_head_attention import RTIDS_Multi_Head_Attention
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class RTIDS_Decoder_Layer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = RTIDS_Norm(d_model)
        self.norm_2 = RTIDS_Norm(d_model)
        self.norm_3 = RTIDS_Norm(d_model)
        self.attn = RTIDS_Multi_Head_Attention(heads, d_model, dropout)
        self.msk_attn = RTIDS_Multi_Head_Attention(heads, d_model, dropout)
        self.feedf = RTIDS_FeedForward(d_model).cuda()
        self.dropout_1 = nn.Dropout(dropout).cuda()
        self.dropout_2 = nn.Dropout(dropout).cuda()
        self.dropout_3 = nn.Dropout(dropout).cuda()

    def forward(self, x, e_outputs, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.msk_attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn(e_outputs,e_outputs,x2,mask=None))
        x2 = self.norm_3(x)
        x = x +self.dropout_3(self.feedf(x2))
        return x

class RTIDS_Decoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout = 0.1):
        super().__init__()
        self.N = N
        self.embed = RTIDS_Embedder(d_model)
        self.pe = RTIDS_Positional_Encoder(d_model,1)
        self.layers = get_clones(RTIDS_Decoder_Layer(d_model, heads, dropout), N)
        self.norm = RTIDS_Norm(d_model)

    def forward(self, trg, e_outputs, mask = None):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x,e_outputs,mask)
        return self.norm(x)
