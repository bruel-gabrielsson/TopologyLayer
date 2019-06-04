#include <torch/extension.h>
#include <vector>

#include "complex.h"

std::vector<torch::Tensor> persistence_forward_hom(SimplicialComplex &X, int MAXDIM);
