#include <torch/extension.h>
#include <vector>

#include "cocycle.h"
#include "interval.h"
#include "complex.h"

// cohomology reduction algorithm
// return barcode

typedef std::map<int,Interval> Barcode;

// perform reduction step on active cocycles Z
// with cocycle x
void reuction_step(const Cocycle &x,\
     std::vector<Cocycle> &Z,\
     Barcode partial_diagram);

std::vector<torch::Tensor> persistence_forward(
    SimplicialComplex &X, torch::Tensor &f, int MAXDIM);

torch::Tensor persistence_backward(
    SimplicialComplex &X, std::vector<torch::Tensor> &grad_res);
