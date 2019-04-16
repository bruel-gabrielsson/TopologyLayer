import torch
import torch.nn as nn
import numpy as np

class SumBarcodeLengths(nn.Module):
    """
    Layer that sums up lengths of barcode in persistence diagram

    ignores infinite bars, and padding
    """
    def __init__(self):
        super(SumBarcodeLengths, self).__init__()

    def forward(self, dgm):
        births, deaths = dgm[:,:,0], dgm[:,:,1]
        lengths = births - deaths
        # remove infinite and irrelevant bars
        lengths[lengths == np.inf] = 0
        lengths[lengths != lengths] = 0
        # return the sum of the barcode lengths
        return torch.sum(lengths, dim=1)
