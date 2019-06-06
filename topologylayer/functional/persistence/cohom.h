#include <torch/extension.h>
#include <vector>

#include "cocycle.h"
// #include "interval.h"
#include "complex.h"

// cohomology reduction algorithm
// return barcode

// typedef std::map<int,Interval> Barcode;

// forward function for any filtration
std::vector<torch::Tensor> persistence_forward(
    SimplicialComplex &X, size_t MAXDIM);

// backward function for lower-star
torch::Tensor persistence_backward(
    SimplicialComplex &X, std::vector<torch::Tensor> &grad_res);

// backward function for flag complexes
torch::Tensor persistence_backward_flag(
     SimplicialComplex &X,
     torch::Tensor &y,
     std::vector<torch::Tensor> &grad_res);
