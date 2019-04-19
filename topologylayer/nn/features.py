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


def get_raw_barcode_lengths(dgminfo):
    """
    get barcode lengths from barcode pairs
    no filtering
    """
    start, end = get_start_end(dgminfo)
    lengths = end - start
    return lengths

def get_barcode_lengths(dgminfo):
    """
    get barcode lengths from barcode pairs
    filter out infinite bars
    """
    lengths = get_raw_barcode_lengths(dgminfo)
    # remove infinite and irrelevant bars
    lengths[lengths == np.inf] = 0
    lengths[lengths != lengths] = 0
    return lengths


class SumBarcodeLengths(nn.Module):
    """
    Layer that sums up lengths of barcode in persistence diagram

    ignores infinite bars, and padding
    """
    def __init__(self):
        super(SumBarcodeLengths, self).__init__()

    def forward(self, dgminfo):
        lengths = get_barcode_lengths(dgminfo)

        # return the sum of the barcode lengths
        return torch.sum(lengths, dim=1)


def pad_k(t, k, pad=0.0):
    """
    zero pad tensor t until dimension along axis is k

    if t has dimension greater than k, truncate
    """
    lt = len(t)
    if lt > k:
        return t[:k]
    if lt < k:
        fillt = torch.tensor(pad * np.ones(k - lt), dtype=t.dtype)
        return torch.cat((out, fillt))
    return t


class TopKBarcodeLengths(nn.Module):
    """
    Layer that returns top k lengths of persistence diagram in dimension

    inputs:
        k - number of lengths
        dim - homology dimension

    ignores infinite bars and padding
    """
    def __init__(self, k, dim):
        super(TopKBarcodeLengths, self).__init__()
        self.k = k
        self.dim = dim

    def forward(self, dgminfo):
        lengths = get_barcode_lengths(dgminfo)
        # just get relevent dimension
        lengths = lengths[self.dim,:]
        # sort lengths
        sortl, indl = torch.sort(lengths, dim=0, descending=True)

        return pad_k(sortl, self.k, 0.0)
