from torch import nn

class RTIDS_Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=None)
    def forward(self, x):
        return self.embed(x)