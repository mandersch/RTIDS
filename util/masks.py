import numpy as np
import torch
from torch.autograd import Variable

def get_mask(size):
    mask_prob = 0.2
    mask = torch.rand((128, 8, size, size)) > mask_prob
    return mask.cuda()