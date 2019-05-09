#include "complex.h"
#include "cocycle.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>

#include <torch/extension.h>
namespace py = pybind11;


void SimplicialComplex::append(std::vector<size_t> &x) {
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

void SimplicialComplex::printFiltration() {
    for (auto i : filtration_perm ){
        py::print(full_function[i]);
        py::print(cells[i]);
    }
}

void SimplicialComplex::initialize() {
    // to call after complex is built.

    // allocate vectors
    filtration.reserve(cells.size());
    filtration_perm.reserve(cells.size());
    backprop_lookup.reserve(cells.size());
    full_function.reserve(cells.size());
    function_map.reserve(cells.size());


	// first build reverse map
    std::map<std::vector<size_t>, size_t> reverse_map;
	int maxdim  = 0;
	int indx = 0;
    for(auto s : cells){
		reverse_map[s] = indx++;
		maxdim = (maxdim < s.size()-1) ? s.size()-1 : maxdim;
	}

	// inialize boundary
	bdr.reserve(indx);


	std::vector<size_t> s_copy; // copy of s
	for (auto s: cells){
		std::vector<size_t> tmp; // holds boundary
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
	    sort(tmp.begin(), tmp.end());

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

// TODO: figure out how to use template with PyBind...
//template <typename T>
//void SimplicialComplex::extend(std::vector<T> &f) {}
void SimplicialComplex::extend(std::vector<double> &f) {
    const size_t N(cells.size());
    full_function.resize(N);
    function_map.resize(N);
    for (size_t i = 0; i < N; i++ ){
        int element = *std::max_element(cells[i].begin(),cells[i].end(),[&f](int i1, int i2){return f[i1]<f[i2];});
        full_function[i] = std::pair<double,int>(f[element], cells[i].size()-1);
        function_map[i] = element;
    }
}

void SimplicialComplex::sortedOrder() {
    filtration_perm.resize(full_function.size());
    std::iota(filtration_perm.begin(), filtration_perm.end(), 0);

    // sort indexes based on comparing values in x - take into account dimension
      std::sort(filtration_perm.begin(), filtration_perm.end(), \
            [this](int i1, int i2) {\
                return (full_function[i1].first==full_function[i2].first) ? full_function[i1].second < full_function[i2].second : full_function[i1].first < full_function[i2].first;});

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
