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

// // reworked extension function
// // input is on vertices
// void Cohomology::extend(const std::vector<double> &f ){
// 	// move this to initialization
// 	const size_t N(complex.size());
// 	for(size_t i = 0 ; i<N;++i){
// 		int element = *std::max_element(complex[i].begin(),complex[i].end(),[&f](int i1, int i2){return f[i1]<f[i2];});
// 	//	std::cout<<"extending function to "<<i<<"  "<<element <<std::endl;
// 		full_function[i] = std::pair<double,int>(f[element], complex[i].size()-1);
// 		function_map[i] = element;
// 	}
//
// }

// // stable sorting - there is one trick we need to take care of -
// // since we do the extension - we need to make sure the
// // returned order is valid
// std::vector<int> Cohomology::sortedOrder(){
// 	std::vector<int> idx(full_function.size());
// 	std::iota(idx.begin(), idx.end(), 0);
//
// 	 // sort indexes based on comparing values in x - take into account dimension
//     	std::sort(idx.begin(), idx.end(), [this](int i1, int i2) {return (full_function[i1].first==full_function[i2].first) ? full_function[i1].second < full_function[i2].second : full_function[i1].first < full_function[i2].first;});
//
// 	return idx;
// }
