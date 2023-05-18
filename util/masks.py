import numpy as np
import torch
from torch.autograd import Variable

def get_mask(size):
    mask = np.triu(np.ones((1,size,size)),k=1).astype("uint8")
    return Variable(torch.from_numpy(mask) == 0)