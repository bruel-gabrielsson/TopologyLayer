#include "complex.h"
#include <iostream>
#include <vector>
#include <algorithm>

#include <torch/extension.h>
namespace py = pybind11;


void SimplicialComplex::append(std::vector<int> &x) {
  // make sure simplex is sorted
  sort(x.begin(), x.end());
  // add to list of cells
  cells.push_back(x);
}

void SimplicialComplex::print() {
  for (auto s : cells) {
    // use python print function
    py::print(s);
  }
}
