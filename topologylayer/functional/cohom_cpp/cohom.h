#include <torch/extension.h>

// cohomology reduction algorithm
// return barcode




/*
DEMO function below
*/

std::vector<at::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell);

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights);

torch::Tensor d_sigmoid(torch::Tensor z);

torch::Tensor d_tanh(torch::Tensor z);

torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0);
