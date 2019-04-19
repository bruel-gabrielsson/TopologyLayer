import torch
import torch.nn as nn
import numpy as np

def get_start_end(dgminfo):
    """
    get start and endpoints of barcode pairs
    input:
        dgminfo - Tuple consisting of diagram tensor and bool
            bool = true if diagram is of sub-level set type
            bool = false if diagram is of super-level set type
    output - start, end tensors of diagram
    """
    dgm, issublevel = dgminfo
    if issublevel:
        # sub-level set filtration e.g. Rips
        start, end = dgm[:,:,0], dgm[:,:,1]
    else:
        # super-level set filtration
        end, start = dgm[:,:,0], dgm[:,:,1]
    return start, end


class SumBarcodeLengths(nn.Module):
    """
    Layer that sums up lengths of barcode in persistence diagram

    ignores infinite bars, and padding
    """
    def __init__(self):
        super(SumBarcodeLengths, self).__init__()

    def forward(self, dgminfo):
        start, end = get_start_end(dgminfo)
        lengths = end - start
        # remove infinite and irrelevant bars
        lengths[lengths == np.inf] = 0
        lengths[lengths != lengths] = 0
        # return the sum of the barcode lengths
        return torch.sum(lengths, dim=1)
