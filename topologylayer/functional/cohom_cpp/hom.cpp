#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <limits>


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
std::vector<Cocyle> sorted_boundary(SimplicialComplex &X, int MAXDIM) {
	// empty boundary matrix
	std::vector<Cocycle> B;
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
		B.emplace_back(Cocycle(j, row_inds));
	}

	return B;
}


/*
Function to reduce boundary matrix.
*/


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

	 // run standard reduction algorithm

	 // fill in diagram

   return diagram;
 }
