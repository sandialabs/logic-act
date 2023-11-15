/* Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
 * Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
 * certain rights in this software.
 * 
 * Written by Jed A. Duersch, Sandia National Laboratories, Livermore, CA.
 * 
 * This algorithm implements logical activation functions according to the paper,
 * "Logical Activation Functions for Training Arbitrary Probabilistic Boolean Operations."
 */


#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#define MAX_ARGS 6
#define CORES_PER_SM 64
#define NUM_SM 108


template <typename scalar_t>
__global__ void logic11_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> X_,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> B_,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> Y_,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dYdX_,
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> I_dB_,
    const int nAr, const int bOut, const int nSl, const int nFa, const int nOut, const int nVe, const int dbs, const int dbf) {
  // Logical activation usage:
  //   X : Argument array, <nSl, nIn=nAr*(nOut/bOut), nFa>
  //   B : Belief table,   <1 or nSl, 2**nAr, nOut, 1 or nFa>
  //   nAr : Number of arguments in each belief function
  //   bOut : Number of outputs for each argument block
  //
  // Output:
  //   Y     : Output array,   <nSl, nOut, nFa>
  //   dYdX  : Derivatives,    <nSl, nAr, nOut, nFa>
  //   I_dB  : Vertex Index,   <nSl, nOut, nFa>
  const int nOutFa = nOut*nFa;
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int stx = blockDim.x * gridDim.x;
  const int nEl = nSl*nOutFa;

  scalar_t b_[1 << MAX_ARGS];
  int i_db_[1 << MAX_ARGS];

  for (int x = ix; x < nEl; x += stx){
    int c = x / nFa;
    int s = x / nOutFa;
    int f = x - c * nFa;
    c -= s*nOut;
    int ci = (c/bOut)*nAr;
    // Slow index for B.
    int sb = s*dbs;
    // Fast index for B.
    int fb = f*dbf;

    // Reset size of temporary belief table.
    int nVe_ = nVe;
    // Copy belief table into temporary array.
    for (int v = 0; v < nVe; v++) {
      b_[v] = B_[sb][v][c][fb];
      i_db_[v] = v;
    } // v++

    // Process each argument
    for (int a = 0; a < nAr; a++) {
      // Get current argument.
      scalar_t x_ = X_[s][ci + a][f];
      scalar_t sx = 1.0;
      if (x_ < 0.0) {
        x_ = -x_;
        sx = -1.0;
      }

      // Process new verticies. This is the number of updates to process.
      nVe_ = nVe_ >> 1;
      // This is the number of temporary elements currently used per entry in b_.
      int nb = 1 + a;
      for (int v = 0; v < nVe_; v++) {
        int v0 = v<<1;
        int v1 = v0 + 1;
        scalar_t sz = 1.0;

        // Switching sign of x requires flipping arguments.
        if (sx < 0.0) {
          v1 = v0;
          v0 = v1 + 1;
        }
        scalar_t b0 = b_[v0*nb];
        scalar_t b1 = b_[v1*nb];

        // Ensure b1 >= b0 by flipping signs. The chain rule will cancel the sign flip in the derivative.
        if (b1 < b0) {
          sz = -1.0;
          b0 = -b0;
          b1 = -b1;
        }

        if (x_ >= b1 - ( b0 < 0.0 ? 0.0 : b0 )) {
          // Copy b1 into update.
          b_[v*(nb + 1)] = b1;
          for (int i = 1; i < nb; i++) {
            b_[v*(nb + 1) + i] = b_[v1*nb + i];
          }
          // Derivative with respect to x.
          b_[v*(nb + 1) + nb] = 0.0;
          // Active argument.
          i_db_[v] = i_db_[v1];
        } else if (b0 < 0.0) {
          // Copy x_ into update.
          b_[v*(nb + 1)] = x_;
          for (int i = 1; i < nb; i++) {
            b_[v*(nb + 1) + i] = 0.0;
          }
          // Derivative with respect to x.
          b_[v*(nb + 1) + nb] = sx*sz;
          // Active argument.
          i_db_[v] = -1;
        } else {
          // Copy x_ + b0 into update.
          b_[v*(nb + 1)] = x_ + b0;
          for (int i = 1; i < nb; i++) {
            b_[v*(nb + 1) + i] = b_[v0*nb + i];
          }
          // Derivative with respect to x.
          b_[v*(nb + 1) + nb] = sx*sz;
          // Active argument.
          i_db_[v] = i_db_[v0];
        }
        // Finally, account for sign flip from vertices.
        b_[v*(nb + 1)] *= sz;
      } // v++
    } // a++
    // We can now copy the final belief result into the output arrays.
    Y_[s][c][f] = b_[0];
    for (int a = 0; a < nAr; a++) {
      dYdX_[s][a][c][f] = b_[1 + a];
    }
    I_dB_[s][c][f] = i_db_[0];
  } // x += stx
}


template <typename scalar_t>
__global__ void logic11_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> dY_,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dYdX_,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> I_dB_,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> dX_,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dB_,
    const int nAr, const int bOut, const int nSl, const int nFa, const int nOut, const int dbs, const int dbf) {
  // Inputs :
  //   dY   : Gradient wrt Y,    <nSl, nOut, nFa>
  //   dYdX : Saved derivatives, <nSl, nAr, nOut, nFa>
  //   i_dB : Saved indices,     <nSl, nOut, nFa>
  //   bOut : Number of outputs for each argument block
  //   nSlB : Number of slow elements in B
  //   nFaB : Number of fast elements in B
  //
  // Outputs:
  //   dX   : Argument grad, <nSl, nIn=nAr*(nOut/bOut), nFa>
  //   dB   : Belief grad,   <nSlB, 2**nAr, nOut, nFaB>

  const int nOutFa = nOut*nFa;
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int stx = blockDim.x * gridDim.x;
  const int nEl = nSl*nOutFa;
  for (int x = ix; x < nEl; x += stx){
    int c = x / nFa;
    int s = x / nOutFa;
    int f = x - c * nFa;
    c -= s*nOut;
    int ci = (c/bOut)*nAr;
    int sb = s*dbs;
    int fb = f*dbf;
    for (int a = 0; a < nAr; a++) {
      atomicAdd(&dX_[s][ci + a][f], dY_[s][c][f]*dYdX_[s][a][c][f]);
    }
    // The derivative of each output only has one nonzero with respect to the original belief vertex.
    // i_db[s][c][f] is the index of that nonzero and the value of the derivative is always 1.
    // If the index is -1, there is no nonzero vertex derivative.
    int i = I_dB_[s][c][f];
    if (i >= 0) {
      atomicAdd(&dB_[sb][i][c][fb], dY_[s][c][f]);
    }
  } // x += stx
}


std::vector<torch::Tensor> logic11_cuda_forward(
    const torch::Tensor X,
    const torch::Tensor B,
    const int nAr, const int bOut) {

  const int nSl = X.size(0);
  const int nFa = X.size(2);
  const int nOut = B.size(2);
  const int nVe = 1 << nAr;

  // Create output arrays.
  auto Y = torch::empty({nSl, nOut, nFa}, torch::TensorOptions().dtype(X.dtype()).device(X.device()));
  auto dYdX = torch::empty({nSl, nAr, nOut, nFa}, torch::TensorOptions().dtype(X.dtype()).device(X.device()));
  auto I_dB = torch::empty({nSl, nOut, nFa}, torch::TensorOptions().dtype(torch::kInt32).device(X.device()));

  assert(nAr <= MAX_ARGS);
 
  // If mode 0 of B is a singleton, broadcast.
  const int dbs = (B.size(0) == 1 ? 0 : 1);
  const int dbf = (B.size(3) == 1 ? 0 : 1);

  const dim3 threadsPerBlock(CORES_PER_SM);
  const dim3 numBlocks(NUM_SM);

  // AT_DISPATCH_ALL_TYPES ???
  AT_DISPATCH_FLOATING_TYPES(X.type(), "logic_forward_cuda", ([&] {
    logic11_cuda_forward_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
        X.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        B.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        Y.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        dYdX.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        I_dB.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        nAr, bOut, nSl, nFa, nOut, nVe, dbs, dbf);
  }));

  return {Y, dYdX, I_dB};
}

std::vector<torch::Tensor> logic11_cuda_backward(
    const torch::Tensor dY,
    const torch::Tensor dYdX,
    const torch::Tensor I_dB,
    const int bOut, const int nSlB, const int nFaB) {

  const int nSl = dYdX.size(0);
  const int nAr = dYdX.size(1);
  const int nOut = dYdX.size(2);
  const int nFa = dYdX.size(3);
  const int nIn = nAr*(nOut/bOut);
  const int nVe = 1 << nAr;
  // If mode 0 of B is a singleton, broadcast.
  const int dbs = (nSlB == 1 ? 0 : 1);
  const int dbf = (nFaB == 1 ? 0 : 1);

  // Create output arrays.
  torch::Tensor dX = torch::zeros({nSl, nIn, nFa}, torch::TensorOptions().dtype(dY.dtype()).device(dY.device()));
  torch::Tensor dB = torch::zeros({nSlB, nVe, nOut, nFaB}, torch::TensorOptions().dtype(dY.dtype()).device(dY.device()));

  const dim3 threadsPerBlock(CORES_PER_SM);
  const dim3 numBlocks(NUM_SM);

  // AT_DISPATCH_ALL_TYPES ???
  AT_DISPATCH_FLOATING_TYPES(dY.type(), "logic_backward_cuda", ([&] {
    logic11_cuda_backward_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
        dY.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        dYdX.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        I_dB.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        dX.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        dB.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        nAr, bOut, nSl, nFa, nOut, dbs, dbf);
  }));

  return {dX, dB};
}



