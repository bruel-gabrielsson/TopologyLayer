
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
 void persistence_forward(SimplicialComplex &X, std::vector<double> f) {

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




torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}


std::vector<at::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);

  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
  auto gates = gate_weights.chunk(3, /*dim=*/1);

  auto input_gate = torch::sigmoid(gates[0]);
  auto output_gate = torch::sigmoid(gates[1]);
  auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

  auto new_cell = old_cell + candidate_cell * input_gate;
  auto new_h = torch::tanh(new_cell) * output_gate;

  return {new_h,
          new_cell,
          input_gate,
          output_gate,
          candidate_cell,
          X,
          gate_weights};
}


// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  auto d_output_gate = torch::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;

  auto gates = gate_weights.chunk(3, /*dim=*/1);
  d_input_gate *= d_sigmoid(gates[0]);
  d_output_gate *= d_sigmoid(gates[1]);
  d_candidate_cell *= d_elu(gates[2]);

  auto d_gates =
      torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gates.mm(weights);
  const auto state_size = grad_h.size(1);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}
