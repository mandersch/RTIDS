import numpy as np
import torch
from torch.autograd import Variable

def get_mask(size):
    mask = np.triu(np.ones((1,8,128,128)),k=1).astype("uint8")
    return Variable(torch.from_numpy(mask) == 0).cuda()