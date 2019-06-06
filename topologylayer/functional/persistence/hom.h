#include <torch/extension.h>
#include <vector>

#include "complex.h"

std::vector<torch::Tensor> persistence_forward_hom(SimplicialComplex &X, size_t MAXDIM, size_t alg_opt);
