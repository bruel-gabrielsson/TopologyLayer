
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <limits>

#include "cohom.h"

void reduction_step(SimplicialComplex &X,\
		 const size_t i,\
     std::vector<Cocycle> &Z,\
     std::vector<torch::Tensor> &diagram,\
		 std::vector<int> &nbars,
	 	 const size_t MAXDIM) {

	 // get cocycle
	 Cocycle c = X.bdr[i];

   // perform single reduction step with cocycle x
   bool flag = false;
   auto pivot = Z.rbegin();
   for(auto x  = Z.rbegin(); x != Z.rend();  ++x){
     // see if inner product is non-zero
     if(x->dot(c) > 0){
       if(flag==false){
         // save as column that will be used for schur complement
         pivot = x;
         flag=true;
       } else {
         // schur complement
         x->add(*pivot);
       }
     }
   }

   // cocycle was closed
   if (flag) {
     // add persistence pair to diagram.

		 // get birth and death indices
		 size_t bindx = pivot->index;
		 size_t dindx = c.index;
		 // get birth dimension
		 size_t hdim = X.dim(bindx);
		 //py::print("bindx: ", bindx, " dindx: ", dindx, " hdim: ", hdim);

		 // delete reduced column from active cocycles
		 // stupid translation from reverse to iterator
		 Z.erase(std::next(pivot).base());

		 // check if we want this bar
		 if (hdim > MAXDIM) { return; }

		 // get location in diagram
		 size_t j = nbars[hdim]++;

		 // put births and deaths in diagram.
		 (diagram[hdim][j].data<float>())[0] = (float) X.full_function[bindx].first;
		 (diagram[hdim][j].data<float>())[1] = (float) X.full_function[dindx].first;

		 // put birth/death indices of bar in X.backprop_lookup
		 X.backprop_lookup[hdim][j] = {(int) bindx, (int) dindx};
   } else {
     //  cocycle opened
		 size_t bindx = c.index;
		 // add active cocycle
     Z.emplace_back(Cocycle(bindx));
   }
 }


/*
	INPUTS:
		X - simplicial complex
			IMPORTANT: assumes that X has been initialized, and filtration has been extended
		MAXDIM - maximum homology dimension
	OUTPUTS: vector of tensors - t
	 t[k] is float32 tensor with barcode for dimension k
*/
std::vector<torch::Tensor> persistence_forward(SimplicialComplex &X, size_t MAXDIM) {

   // produce sort permutation on X
   X.sortedOrder();

   // empty vector of active cocycles
   std::vector<Cocycle> Z;

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
	 // keep track of how many pairs we've put in diagram
	 std::vector<int> nbars(MAXDIM+1);
	 for (size_t k = 0; k < MAXDIM+1; k++) {
		 nbars[k] = 0;
	 }

	 // go through reduction algorithm
   for (size_t i : X.filtration_perm ) {
     reduction_step(X, i, Z, diagram, nbars, MAXDIM);
   }

	 // add infinite bars using removing columns in Z
	 // backprop_lookup death index = -1,
	 // death time is std::numeric_limits<float>::infinity()
	 // while (!(Z.empty())){
	 for (auto pivot  = Z.begin(); pivot != Z.end();  ++pivot) {
		 // Cocycle pivot = Z.pop_back();
		 // get birth index
		 size_t bindx = pivot->index;
		 // get birth dimension
		 size_t hdim = X.bdr[bindx].dim();
		 if (hdim > MAXDIM) { continue; }

		 // get location in diagram
		 size_t j = nbars[hdim]++;

		 // put births and deaths in diagram.
		 (diagram[hdim][j].data<float>())[0] = (float) X.full_function[bindx].first;
		 (diagram[hdim][j].data<float>())[1] = (float) std::numeric_limits<float>::infinity();

		 // put birth/death indices of bar in X.backprop_lookup
		 X.backprop_lookup[hdim][j] = {(int) bindx, -1};
	 }


   return diagram;
 }


/*
INPUTS:
		X - simplicial complex
		IMPORTANT: assumes that X has been initialized
	grad_res - vector of vectors of tensors
	same as input format:
	grad_res[k] is float32 tensor of gradient of births/deaths in dimension k
OUTPUT:
	grad_f - gradient w.r.t. original function
*/
torch::Tensor persistence_backward(
 SimplicialComplex &X, std::vector<torch::Tensor> &grad_res) {

	 int N = X.ncells[0]; // number of cells in X
	 torch::Tensor grad_f = torch::zeros({N},
		 torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided));
	// pointer to data
	float *grad_f_data = grad_f.data<float>();

	int NDIMS = grad_res.size();

	// loop over homology dimensions
	for (int k = 0; k < NDIMS; k++) {

		// number of bars in dimension k
		int Nk = grad_res[k].size(0);


		// loop over bars in dim k barcode
		for (int j = 0; j < Nk; j++) {
			// get pointers to filtration indices and pointers
			// int *filtind = grad_res[k][j].data<int>();
			float *grad = grad_res[k][j].data<float>();

			int bi = X.backprop_lookup[k][j][0];
			// check for non-infinite bar
			if (bi != -1) {
				// get birth cell
				// auto ci = X.filtration_perm[bi];
				// find critical vertex
				auto i = X.function_map[bi][0];
				// add gradient to critical vertex.
				grad_f_data[i] += grad[0];
			}

			int di = X.backprop_lookup[k][j][1];
			// check for non-infinite bar
			if (di != -1) {
				// get death cell
				// auto ci = X.filtration_perm[di];
				// find critical vertex
				auto i = X.function_map[di][0];
				// add gradient to critical vertex.
				grad_f_data[i] += grad[1];
			}
		}
	}

	 return grad_f;
}


/*
INPUTS:
	X - simplicial complex
		IMPORTANT: assumes that X has been initialized
	y - coordinate positions
	grad_res - vector of vectors of tensors
		same as input format:
		grad_res[k] is float32 tensor of gradient of births/deaths in dimension k
OUTPUT:
	grad_y - gradient of coordinate positions y
*/
torch::Tensor persistence_backward_flag(
 SimplicialComplex &X,
 torch::Tensor &y,
 std::vector<torch::Tensor> &grad_res) {

	 int N = y.size(0); // number of points
	 int D = y.size(1);
	 torch::Tensor grad_y = torch::zeros({N, D},
		 torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided));
	// pointer to data
	//float *grad_y_data = grad_y.data<float>();

	int NDIMS = grad_res.size();

	// loop over homology dimensions
	for (int k = 0; k < NDIMS; k++) {

		// number of bars in dimension k
		int Nk = grad_res[k].size(0);


		// loop over bars in dim k barcode
		for (int j = 0; j < Nk; j++) {
			// get pointers to filtration indices and pointers
			// int *filtind = grad_res[k][j].data<int>();
			float *grad = grad_res[k][j].data<float>();

			// get index of birth
			int bi = X.backprop_lookup[k][j][0];
			// check for non-infinite bar
			if (bi != -1) {
				// check that birth dim is > 0
				if (X.full_function[bi].second > 0) {
					// get birth cell
					// find critical edge
					auto edge = X.function_map[bi] ;
					// produce unit vector along edge
					torch::Tensor dy = y[edge[0]] - y[edge[1]];
					dy /= torch::norm(dy);
					// add gradient to critical vertex.
					grad_y[edge[0]] += grad[0] * dy;
					grad_y[edge[1]] -= grad[0] * dy;
				}
			}

			int di = X.backprop_lookup[k][j][1];
			// check for non-infinite bar
			if (di != -1) {
				// get death cell
				// auto ci = X.filtration_perm[di];
				// find critical vertex
				auto edge = X.function_map[di];
				// produce unit vector along edge
				torch::Tensor dy = y[edge[0]] - y[edge[1]];
				// TODO: check for zero norm.
				dy /= torch::norm(dy);
				// add gradient to critical vertex.
				grad_y[edge[0]] += grad[1] * dy;
				grad_y[edge[1]] -= grad[1] * dy;
			}
		}
	}

	 return grad_y;
}
