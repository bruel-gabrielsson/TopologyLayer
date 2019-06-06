#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <map>
#include "sparsevec.h"

#include "hom.h"


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
std::vector<SparseF2Vec<int>> sorted_boundary(SimplicialComplex &X, size_t MAXDIM) {
	// empty boundary matrix
	std::vector<SparseF2Vec<int>> B;
	// build boundary in sorted order using filtration_perm
	// should also use filtration_perm to permute nzs in rows of columns
	std::vector<int> row_inds; // row indices for column
	for (size_t j : X.filtration_perm ) {
		//if (X.dim(j) > MAXDIM+2) { continue; }
		row_inds.clear(); // clear out column
		// go through non-zeros in boundary
		for (auto i : X.bdr[j].cochain.nzinds) {
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
		// if nnz = 0, the reduction is complete
		while (B[j].nnz() > 0) {
			int piv = B[j].last();
			if (pivot_to_col.count(piv) > 0) {
				int k = pivot_to_col[piv];
				// there is a column with that pivot
				B[j].add(B[k]);
			} else {
				// there is no column with that pivot
				pivot_to_col[piv] = j;
				break;
			}
		} // end column reduction
	} // end for loop
	return;
}

/*
Function to reduce boundary matrix.
Tries to minimize nnz by continuing to reduce past pivot
INPUT:
	B - boundary matrix
	pivot_to_col - map from pivot to column
OUTPUT:
	none - inputs are modified in-place.
*/
void homology_reduction_alg2(std::vector<SparseF2Vec<int>> &B, std::map<int, int> &pivot_to_col) {
	// loop over columns of boundary matrix
	for (size_t j = 0; j < B.size(); j++) {
		// if nnz = 0, the reduction is complete
		size_t offset = 0; // position from last
		while (B[j].nnz() > offset) {
			int piv = B[j].from_end(offset);
			if (pivot_to_col.count(piv) > 0) {
				int k = pivot_to_col[piv];
				// there is a column with that pivot
				B[j].add(B[k]);
			} else {
				// there is no column with that pivot

				// see if we've found new pivot
				if (offset == 0) {pivot_to_col[piv] = j;}

				// increase offset
				offset++;
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
		alg_opt - 0 - standard reduction algorithm
							1 - nz minimizing reduction algorithm
	OUTPUTS: vector of tensors - t
	 t[k] is float32 tensor with barcode for dimension k
*/
std::vector<torch::Tensor> persistence_forward_hom(SimplicialComplex &X, size_t MAXDIM, size_t alg_opt) {

   // produce sort permutation on X
   X.sortedOrder();

	 // initialize reutrn diagram
	 std::vector<torch::Tensor> diagram(MAXDIM+1); // return array
	 for (size_t k = 0; k < MAXDIM+1; k++) {
		 int Nk = X.numPairs(k); // number of bars in dimension k
		 // allocate return tensor
		 diagram[k] = torch::empty({Nk,2},
			 torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).requires_grad(true));
		 // allocate critical indices
		 // TODO: do this in intialization since number of pairs is deterministic
		 X.backprop_lookup[k].resize(Nk);
	 }

	 // produce boundary matrix
	 std::vector<SparseF2Vec<int>> B = sorted_boundary(X, MAXDIM);

	 // run standard reduction algorithm
	 std::map<int, int> pivot_to_col;
	 if (alg_opt == 0) {
		 // standard reduction algorithm
		 homology_reduction_alg(B, pivot_to_col);
	 } else {
		 // modified reduction algorithm
		 homology_reduction_alg2(B, pivot_to_col);
	 }


	 // keep track of how many pairs we've put in diagram
	 std::vector<size_t> nbars(MAXDIM+1);
	 for (size_t k = 0; k < MAXDIM+1; k++) {
		 nbars[k] = 0;
	 }

	 // fill in diagram
	 for (size_t j = 0; j < B.size(); j++ ) {
		 // see if column j is completely reduced

		 if (B[j].nnz() == 0) {
			 // homology born in column j

			 // in filtration order:
			 // birth at j, death at k = pivot_to_col[j]
			 size_t bindx = X.filtration_perm[j];
			 size_t hdim = X.dim(bindx);
			 if (hdim > MAXDIM) { continue; }
			 // get location in diagram
			 size_t di = nbars[hdim]++;
			 // set birth time
			 (diagram[hdim][di].data<float>())[0] = (float) X.full_function[bindx].first;

			 if (pivot_to_col.count(j) > 0) {
				 // there is a finite death.
				 int k = pivot_to_col[j]; // column that has j as pivot
				 size_t dindx = X.filtration_perm[k];
				 // get dimension of cell with filtration poisition j
				 (diagram[hdim][di].data<float>())[1] = (float) X.full_function[dindx].first;
				 // need to fill in X.backprop_lookup as well
				 X.backprop_lookup[hdim][di] = {(int) bindx, (int) dindx};
			 } else {
				 // infinite death
				 (diagram[hdim][di].data<float>())[1] = std::numeric_limits<float>::infinity();
				 X.backprop_lookup[hdim][di] = {(int) bindx, -1};
			 }
		 }
	 }

   return diagram;
 }
