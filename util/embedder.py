from torch import nn
import torch 

class RTIDS_Embedder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_m = d_model
    def forward(self, x):
        tensor_size = x.size()
        reshaped_x = torch.unsqueeze(x, dim=2)
        # Pad the tensor
        target_size = (tensor_size[0], tensor_size[1], self.d_m)
        padding = (0,target_size[2] - reshaped_x.size(2),0,0,0,0)
        padded_tensor = nn.functional.pad(reshaped_x, padding, "constant", 0)

        return padded_tensor