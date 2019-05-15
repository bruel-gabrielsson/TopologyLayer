
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <limits>

#include "cohom.h"

void reduction_step(const Cocycle &c,\
     std::vector<Cocycle> &Z,\
     Barcode &partial_diagram) {

   // perform single reduction step with cocycle x
   bool flag = false;
   auto pivot = Z.rbegin();
   for(auto x  = Z.rbegin(); x != Z.rend();  ++x){
     // see if inner product is non-zero
     if(x->dot(c)){
       if(flag==false){
         // save as column that will be used for schur complement
         pivot = x;
         flag=true;
       }
       else{
         // schur complement
         x->add(*pivot);
       }
     }
   }

   // cocycle was closed
   if(flag){
     // add assertion that it exists
     partial_diagram[pivot->index].close(c.index);
     // stupid translation from reverse to iterator
     Z.erase(std::next(pivot).base());
   } else {
     //  cocycle opened
     // TODO: check indexing!
     auto i = c.index;
     partial_diagram[i] = Interval(i);
     Z.emplace_back(Cocycle(i));
   }
 }


/*
	INPUTS:
		X - simplicial complex
			IMPORTANT: assumes that X has been initialized
		f - filtration
		MAXDIM - maximum homology dimension
	OUTPUTS: vector of tensors - t
	 t[k] is float32 tensor with barcode for dimension k
*/
std::vector<torch::Tensor> persistence_forward(SimplicialComplex &X, torch::Tensor &f, int MAXDIM) {

   // extend filtration
   X.extend(f);
   // produce sort permutation on X
   X.sortedOrder();

   // empty vector of active cocycles
   std::vector<Cocycle> Z;

   // to store barcode
   Barcode partial_diagram;

   for (auto i : X.filtration_perm ) {
     reduction_step(X.bdr[i], Z, partial_diagram);
   }

   // return things!

	 // old way of doing things
	 std::map<int,std::vector<Interval>> persistence_diagram;
	 // fill in barcode - this will be changed to a tensor
		for(auto it = partial_diagram.begin(); it!=partial_diagram.end(); ++it){
			int  bindx = it->first;
			auto I = it->second;
			if(I.death_index==-1){
				persistence_diagram[X.bdr[bindx].dim()].emplace_back(
					Interval(I.birth_index, I.death_index, X.full_function[I.birth_index].first,std::numeric_limits<float>::infinity()));
			}
			else{
				persistence_diagram[X.bdr[bindx].dim()].emplace_back(
					Interval(I.birth_index, I.death_index, X.full_function[I.birth_index].first,X.full_function[I.death_index].first));
			}
		}

		// return type will be a list of lists of tensors
 	 // t[k] is list of two tensors - homology in dimension k
 	 // t[k][0] is float32 tensor with barcode
 	 // t[k][1] is int32 tensor with critical simplices indices

		// convert output to tensors
		std::vector<torch::Tensor> ret(MAXDIM+1); // return array
		for (int k = 0; k < MAXDIM+1; k++) {
			// TODO: can figure this out directly from X.ncells
			int Nk = persistence_diagram[k].size(); // number of bars in dimension k
			// allocate return tensor
			ret[k] = torch::empty({Nk,2},
				torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).requires_grad(true));
			// allocate critical indices
			X.backprop_lookup[k].resize(Nk);
			// put barcode in return tensor and save critical indices
			for (int j = 0; j < Nk; j++) {
				// birth-death values
				(ret[k][j].data<float>())[0] = (float) persistence_diagram[k][j].birth;
				(ret[k][j].data<float>())[1] = (float) persistence_diagram[k][j].death;
				// birth-death indices
				X.backprop_lookup[k][j] = {persistence_diagram[k][j].birth_index,
																		persistence_diagram[k][j].death_index};
			}
		}

   return ret;
 }


/*
INPUTS:
		X - simplicial complex
		IMPORTANT: assumes that X has been initialized
	grad_res - vector of vectors of tensors
	same as input format:
	grad_res[k] is vector of two tensors
	grad_res[k][0] is float32 tensor of gradient of births/deaths in dimension k
	grad_res[k][1] is same int32 tensor of critical simplex indices
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
				auto ci = X.filtration_perm[bi];
				// find critical vertex
				auto i = X.function_map[ci];
				// add gradient to critical vertex.
				grad_f_data[i] += grad[0];
			}

			int di = X.backprop_lookup[k][j][1];
			// check for non-infinite bar
			if (di != -1) {
				// get death cell
				auto ci = X.filtration_perm[di];
				// find critical vertex
				auto i = X.function_map[ci];
				// add gradient to critical vertex.
				grad_f_data[i] += grad[1];
			}
		}
	}

	 return grad_f;

}
