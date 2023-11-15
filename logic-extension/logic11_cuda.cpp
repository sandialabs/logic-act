/* Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
 * Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
 * certain rights in this software.
 *
 * Written by Jed A. Duersch, Sandia National Laboratories, Livermore, CA.
 *
 * This algorithm implements logical activation functions according to the paper,
 * "Logical Activation Functions for Training Arbitrary Probabilistic Boolean Operations."
 * */

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> logic11_cuda_forward(
    const torch::Tensor X,
    const torch::Tensor B,
    const int nAr, const int bOut);
// returns {Y, dYdX, I_dB}

std::vector<torch::Tensor> logic11_cuda_backward(
    const torch::Tensor dY,
    const torch::Tensor dYdX,
    const torch::Tensor I_dB,
    const int bOut, const int nSlB, const int nFaB);
// returns {dX, dB}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> logic11_forward(
    const torch::Tensor X,
    const torch::Tensor B,
    const int nAr, const int bOut) {
  CHECK_INPUT(X);
  CHECK_INPUT(B);
  return logic11_cuda_forward(X, B, nAr, bOut);
}

std::vector<torch::Tensor> logic11_backward(
    const torch::Tensor dY,
    const torch::Tensor dYdX,
    const torch::Tensor I_dB,
    const int bOut, const int nSlB, const int nFaB) {
  CHECK_INPUT(dY);
  CHECK_INPUT(dYdX);
  CHECK_INPUT(I_dB);
  return logic11_cuda_backward(dY, dYdX, I_dB, bOut, nSlB, nFaB);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &logic11_forward, "Logic11 forward (CUDA)");
  m.def("backward", &logic11_backward, "Logic11 backward (CUDA)");
}


    
