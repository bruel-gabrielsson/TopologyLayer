
#include <torch/extension.h>
#include <iostream>
#include <vector>

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

// IMPORTANT: assumes that X has been initialized
/*
	INPUTS:
		X - simplicial complex
		f - filtration
		MAXDIM - maximum homology dimension
*/
 std::vector<std::vector<torch::Tensor>> persistence_forward(SimplicialComplex &X, torch::Tensor f, int MAXDIM) {

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
	 float *f2 = f.data<float>(); // pointer to data
	 // fill in barcode - this will be changed to a tensor
		for(auto it = partial_diagram.begin(); it!=partial_diagram.end(); ++it){
			int  bindx = it->first;
			auto I = it->second;
			if(I.death_index==-1){
				persistence_diagram[X.bdr[bindx].dim()].emplace_back(Interval(I.birth_index, I.death_index, f2[I.birth_index],-1));
			}
			else{
				persistence_diagram[X.bdr[bindx].dim()].emplace_back(Interval(I.birth_index, I.death_index, f2[I.birth_index],f2[I.death_index]));
			}
		}

		// return type will be a list of lists of tensors
 	 // t[k] is list of two tensors - homology in dimension k
 	 // t[k][0] is float32 tensor with barcode
 	 // t[k][1] is int32 tensor with critical simplices indices

		// convert output to tensors
		std::vector<std::vector<torch::Tensor>> ret(MAXDIM+1); // return array
		for (int k = 0; k < MAXDIM+1; k++) {
			ret[k] = std::vector<torch::Tensor>(2);
			int Nk = persistence_diagram[k].size(); // number of bars in dimension k
			ret[k][0] = torch::empty({Nk,2},
				torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).requires_grad(true));
			ret[k][1] = torch::empty({Nk,2},
				torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).requires_grad(false));
			// put barcode in return tensors
			for (int j = 0; j < Nk; j++) {
				// birth-death values
				(ret[k][0][j].data<float>())[0] = (float) persistence_diagram[k][j].birth;
				(ret[k][0][j].data<float>())[1] = (float) persistence_diagram[k][j].death;
				// birth-death indices
				(ret[k][1][j].data<int>())[0] = (int) persistence_diagram[k][j].birth_index;
				(ret[k][1][j].data<int>())[1] = (int) persistence_diagram[k][j].death_index;
			}
		}


   return ret;
 }
