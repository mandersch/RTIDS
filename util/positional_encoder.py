import torch
from torch import nn
from torch.autograd import Variable
from math import cos, sin, sqrt

class RTIDS_Positional_Encoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 78):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0,d_model,2):
                pe[pos,i] = sin(pos / (1000**(i/d_model)))
                pe[pos,i+1] = cos(pos / (1000**(i/d_model)))
        
        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        # print(x.size())
        # print("PE-size: ", self.pe.size())
        y = Variable(self.pe[:,:seq_len],requires_grad=False).cuda()
        # print("Y-Var size: ",y.size())
        # print(y.size())
        x = x + y #.cuda()
        return x