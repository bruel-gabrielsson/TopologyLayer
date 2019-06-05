#include "complex.h"
#include "cocycle.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>

#include <torch/extension.h>
namespace py = pybind11;


void SimplicialComplex::append(std::vector<int> &x) {
  // make sure simplex is sorted
  sort(x.begin(), x.end());
  // add to list of cells
  cells.push_back(x);
}

void SimplicialComplex::printComplex() {
  for (auto s : cells) {
    // use python print function
    py::print(s);
  }
}

void SimplicialComplex::printDims() {
	for (auto i : ncells) {
		py::print(i);
	}
}

void SimplicialComplex::printCritInds() {
	for (auto dgm : backprop_lookup){
		for (auto v : dgm) {
			py::print(v);
		}
	}
}

void SimplicialComplex::printFunctionMap() {
	for (size_t i = 0; i < function_map.size(); i++) {
		py::print(i, function_map[i]);
	}
}

void SimplicialComplex::printFiltration() {
    for (auto i : filtration_perm ){
        py::print(full_function[i]);
        py::print(cells[i]);
    }
}

// return dimension of cell j
size_t SimplicialComplex::dim(size_t j) {
	return cells[j].size() - 1;
	//return X.bdr[bindx].dim()
}

void SimplicialComplex::initialize() {
    // to call after complex is built.

    // allocate vectors
    filtration_perm.reserve(cells.size());
    full_function.reserve(cells.size());
    function_map.reserve(cells.size());


	// first build reverse map
  std::map<std::vector<int>, size_t> reverse_map;
	size_t maxdim  = 0;
	size_t indx = 0;
  for(auto s : cells){
		reverse_map[s] = indx++;
		size_t sdim = s.size()-1;
		maxdim = (maxdim < sdim) ? sdim : maxdim;
		// make sure ncells is large enough
		while (ncells.size() < maxdim+1) {
			ncells.push_back(0);
		}
		++ncells[sdim]; // increment ncells in appropriate dimension
	}

	// allocate backprop_lookup
	backprop_lookup.resize(maxdim);
	// TODO: if we know how many pairs there are, we can pre-allocate everything

	// inialize boundary
	bdr.reserve(indx);


	std::vector<int> s_copy; // copy of s
	for (auto s: cells){
		std::vector<int> tmp; // holds boundary
		s_copy.clear(); // clear out copy
		if(s.size()>1){
			for (size_t i = 0; i < s.size(); i++ ){
				// copy all of s except for ith element
				s_copy.clear(); // clear out copy
				for (size_t j = 0; j < i; j++ ){
					s_copy.push_back(s[j]);
				}
				for (size_t j = i+1; j < s.size(); j++){
					s_copy.push_back(s[j]);
				}
				// push to boundary matrix
				tmp.push_back(reverse_map[s_copy]);
			}
		}

		// make sure simplex is sorted
	  std::sort(tmp.begin(), tmp.end());

		// reverse_map[s] is index of cell
		// tmp is boundary of s
		bdr.emplace_back(Cocycle(reverse_map[s],tmp));
	}

}

void SimplicialComplex::printBoundary() {
	for (auto c : bdr) {
		c.print();
	}
}

// in dim 0, number of bars is number of vertices
// in dim k, number of bars is number that don't kill something one dim down.
// in dim 1, need to account for infinite dim0 bar
int SimplicialComplex::numPairs(int dim) {
	return (dim == 0) ? ncells[0] : ncells[dim] - this->numPairs(dim-1) + ((dim == 1) ? 1 : 0);
}


//template <typename T>
//void SimplicialComplex::extend(std::vector<T> &f) {}
void SimplicialComplex::extend(torch::Tensor f) {
    const size_t N(cells.size());
    full_function.resize(N);
    function_map.resize(N);
		float *f2 = f.data<float>(); // pointer to data
    for (size_t i = 0; i < N; i++ ){
        int element = *std::max_element(cells[i].begin(),cells[i].end(),[&f2](int i1, int i2){return f2[i1]<f2[i2];}); // < if lower
        full_function[i] = std::pair<float,int>(f2[element], cells[i].size()-1);
        function_map[i] = {element};
    }
}


// extend filtration on 1-cells to filtraiton on all cells
// input:
// 			x - N x dim tensor of positions
void SimplicialComplex::extend_flag(torch::Tensor x) {
	const size_t N(cells.size());
	full_function.resize(N);
	function_map.resize(N);

	for (size_t i = 0; i < N; i++) {

		if (cells[i].size() == 1) {
			// if 0-dim cell, filtration time is 0
			full_function[i] = std::pair<float, int>((float) 0.0, 0);
			function_map[i] = {0};
		} else {
			// if higher-dim simplex, filtration time is latest edge
			float max_f = 0.0;
			for (auto it1 = cells[i].begin(); it1 < cells[i].end(); ++it1) {
				for (auto it2 = cells[i].begin(); it2 < it1; ++it2) {
					float d12 = *(torch::norm(x[*it1] - x[*it2]).data<float>());
					if (d12 > max_f) {
						max_f = d12;
						function_map[i] = {*it2, *it1}; // sorted order
					}
				}
				full_function[i] = std::pair<float,int>(max_f, cells[i].size()-1);
			}
		}

	}
}


void SimplicialComplex::sortedOrder() {
    filtration_perm.resize(full_function.size());
    std::iota(filtration_perm.begin(), filtration_perm.end(), 0);

    // sort indexes based on comparing values in x - take into account dimension
    std::sort(filtration_perm.begin(), filtration_perm.end(), \
        [this](int i1, int i2) {\
            return (full_function[i1].first==full_function[i2].first) ? full_function[i1].second < full_function[i2].second : full_function[i1].first < full_function[i2].first;});

		// fill inverse filtration perm
		inv_filtration_perm.resize(full_function.size());
		for (size_t i = 0; i < filtration_perm.size(); i++) {
			inv_filtration_perm[filtration_perm[i]] = i;
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
