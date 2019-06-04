#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <map>
#include "sparsevec.h"


/*
Function to build boundary of simplicial complex once sorted order is determined.
Just do single boundary matrix.  No blocking by dimension.
column major order
INPUTS:
	X - simplicial complex
		IMPORTANT: assumes that X has been initialized, and filtration has been extended
	MAXDIM - maximum homology dimension
OUTPUTS: boundary matrix in List of Lists format.
*/
std::vector<SparseF2Vec<int>> sorted_boundary(SimplicialComplex &X) {
	// empty boundary matrix
	std::vector<SparseF2Vec<int>> B;
	// build boundary in sorted order using filtration_perm
	// should also use filtration_perm to permute nzs in rows of columns
	std::vector<int> row_inds; // row indices for column
	for (size_t j : X.filtration_perm ) {
		row_inds.clear(); // clear out column
		// go through non-zeros in boundary
		for (auto i : X.bdr[j].cochain) {
			row_inds.push_back(X.inv_filtration_perm[i]);  // push location of bounday cell in filtration
		}
		// append sorted row_inds to B
		sort(row_inds.begin(), row_inds.end());
		B.emplace_back(SparseF2Vec<int>(row_inds));
	}

	return B;
}

/*
Function to reduce boundary matrix.
INPUT:
	B - boundary matrix
	pivot_to_col - map from pivot to column
OUTPUT:
	none - inputs are modified in-place.
*/
void homology_reduction_alg(std::vector<SparseF2Vec<int>> &B, std::map<int, int> &pivot_to_col) {
	// loop over columns of boundary matrix
	for (size_t j = 0; j < B.size(); j++) {
		while (true) {
			int piv = B[j].pivot();
			if (piv == std::numeric_limits<int>::max()) {
				// we have completely reduced column
				break;
			} else {
				if (pivot_to_col.count(piv)) {
					// there is a column with that pivot
					B[j].add(B[piv]);
				} else {
					// there is no column with that pivot
					pivot_to_col[piv] = j;
					break;
				}
			}
		} // end column reduction
	} // end for loop
	return;
}


/*
	Standard reduction algorithm on simplicial complex.
	INPUTS:
		X - simplicial complex
			IMPORTANT: assumes that X has been initialized, and filtration has been extended
		MAXDIM - maximum homology dimension
	OUTPUTS: vector of tensors - t
	 t[k] is float32 tensor with barcode for dimension k
*/
std::vector<torch::Tensor> persistence_forward_hom(SimplicialComplex &X, int MAXDIM) {

   // produce sort permutation on X
   X.sortedOrder();

	 // initialize reutrn diagram
	 std::vector<torch::Tensor> diagram(MAXDIM+1); // return array
	 for (int k = 0; k < MAXDIM+1; k++) {
		 int Nk = X.numPairs(k); // number of bars in dimension k
		 // allocate return tensor
		 diagram[k] = torch::empty({Nk,2},
			 torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).requires_grad(true));
		 // allocate critical indices
		 // TODO: do this in intialization since number of pairs is deterministic
		 X.backprop_lookup[k].resize(Nk);
	 }
	 // keep track of how many pairs we've put in diagram
	 std::vector<int> nbars(MAXDIM+1);
	 for (int k = 0; k < MAXDIM+1; k++) {
		 nbars[k] = 0;
	 }

	 // produce boundary matrix
	 std::vector<SparseF2Vec<int>> B = sorted_boundary(X);

	 // run standard reduction algorithm
	 std::map<int, int> pivot_to_col;
	 homology_reduction_alg(B, pivot_to_col);

	 // fill in diagram

   return diagram;
 }
