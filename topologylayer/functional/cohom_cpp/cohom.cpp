
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
 void persistence_forward(SimplicialComplex &X, torch::Tensor f) {

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

   // TODO: return things!

   return;
 }
