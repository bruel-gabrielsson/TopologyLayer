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
        start, end = dgm[:,0], dgm[:,1]
    else:
        # super-level set filtration
        end, start = dgm[:,0], dgm[:,1]
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
        return torch.sum(lengths, dim=0)


def get_barcode_lengths_means(dgminfo):
    """
    return lengths and means of barcode

    set irrelevant or infinite to zero
    """
    start, end = get_start_end(dgminfo)
    lengths = end - start
    means = (end + start)/2
    # remove infinite and irrelvant bars
    means[lengths == np.inf] = 0 # note this depends on lengths
    means[lengths != lengths] = 0
    lengths[lengths == np.inf] = 0
    lengths[lengths != lengths] = 0
    return lengths, means


class BarcodePolyFeature(nn.Module):
    """
    applies function
    sum length^a * mean^b
    over lengths and means of barcode
    """
    def __init__(self, dim, a, b):
        super(BarcodePolyFeature, self).__init__()
        self.dim = dim
        self.a   = a
        self.b   = b

    def forward(self, dgminfo):
        lengths, means = get_barcode_lengths_means(dgminfo)
        lengths = lengths[self.dim]
        means = means[self.dim]
        return torch.sum(torch.mul(torch.pow(lengths, self.a), torch.pow(means, self.b)))


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
        return torch.cat((t, fillt))
    return t


class TopKBarcodeLengths(nn.Module):
    """
    Layer that returns top k lengths of persistence diagram in dimension

    inputs:
        dim - homology dimension
        k - number of lengths

    ignores infinite bars and padding
    """
    def __init__(self, dim, k):
        super(TopKBarcodeLengths, self).__init__()
        self.k = k
        self.dim = dim

    def forward(self, dgminfo):
        lengths = get_barcode_lengths(dgminfo)
        # just get relevent dimension
        lengths = lengths[self.dim]
        # sort lengths
        sortl, indl = torch.sort(lengths, dim=0, descending=True)

        return pad_k(sortl, self.k, 0.0)


class PartialSumBarcodeLengths(nn.Module):
    """
    Layer that computes a partial sum of barckode lengths

    inputs:
        dim - homology dimension
        skip - skip this number of the longest bars

    ignores infinite bars and padding
    """
    def __init__(self, dim, skip):
        super(PartialSumBarcodeLengths, self).__init__()
        self.skip = skip
        self.dim = dim

    def forward(self, dgminfo):
        lengths = get_barcode_lengths(dgminfo)
        # just get relevent dimension
        lengths = lengths[self.dim]
        # sort lengths
        sortl, indl = torch.sort(lengths, dim=0, descending=True)

        return torch.sum(sortl[self.skip:])
