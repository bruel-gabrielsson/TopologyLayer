#include <torch/extension.h>
#include <vector>

#include "cocycle.h"
// #include "interval.h"
#include "complex.h"

// cohomology reduction algorithm
// return barcode

// typedef std::map<int,Interval> Barcode;

// perform reduction step on active cocycles Z
// with cocycle x
void reuction_step(SimplicialComplex &X,\
    const size_t i,\
    std::vector<Cocycle> &Z,\
    std::vector<torch::Tensor> &diagram,\
 	std::vector<int> &nbars);

// forward function for any filtration
std::vector<torch::Tensor> persistence_forward(
    SimplicialComplex &X, int MAXDIM);

// backward function for lower-star
torch::Tensor persistence_backward(
    SimplicialComplex &X, std::vector<torch::Tensor> &grad_res);

// backward function for flag complexes
torch::Tensor persistence_backward_flag(
     SimplicialComplex &X,
     torch::Tensor &y,
     std::vector<torch::Tensor> &grad_res);
