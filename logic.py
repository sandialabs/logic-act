''' Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
    Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
    certain rights in this software.

    Written by Jed A. Duersch, Sandia National Laboratories, Livermore, CA.

    This algorithm implements logic activation functions from the paper
    "Logical Activation Functions for Training Arbitrary Probabilistic Boolean Operations."
'''

import math
import torch

import logic11_cpp
import logic11_cuda
from torch import Tensor

__all__ = ['Logic']

class LogicFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, B, nAr, bOut):
        ''' Usage:
           X : Arguments,    <nSlow, nIn = nAr*(nOut/bOut), nFast>
           B : Belief table, <nSlow or 1, 2**nAr, nOut, nFast or 1>
        
        Saved for backward:
           dYdX : Float32 <nSlow, nAr, nOut, nFast>
           I_dB : Int32   <nSlow, nOut, nFast>
        '''
        if X.device == torch.device('cpu'):
            outputs = logic11_cpp.forward(X, B, nAr, bOut)
        else:
            outputs = logic11_cuda.forward(X, B, nAr, bOut)
        Y = outputs[0]
        # dYdX, I_dB
        ctx.save_for_backward(*outputs[1:])
        ctx.bOut = bOut
        ctx.nSlB = B.size(0)
        ctx.nFaB = B.size(3)
        return Y

    @staticmethod
    def backward(ctx, dY):
        if dY.device == torch.device('cpu'):
            outputs = logic11_cpp.backward(dY, *ctx.saved_tensors, ctx.bOut, ctx.nSlB, ctx.nFaB)
        else:
            outputs = logic11_cuda.backward(dY.contiguous(), *ctx.saved_tensors, ctx.bOut, ctx.nSlB, ctx.nFaB)
        dX, dB = outputs
        return dX, dB, None, None


class Logic(torch.nn.Module):
    """ Logical activation functions. """

    __constants__ = ['num_arg', 'mode_in', 'num_out', 'b_out', 'num_in', 'H']
    num_arg: int
    mode_in: int
    num_in: int
    num_out: int
    b_out: int
    num_in: int
    H: Tensor

    def __init__(self,
                 num_arg: int,
                 mode_in: int,
                 num_in: int,
                 num_out: int,
                 device=None,
                 dtype=None) -> None:
        ''' Usage:
           num_arg : The number of inputs per activation output. Inputs are partitioned into blocks of this size.
           mode_in : This is the input of the input tensor that is used for parititioning logic blocks. The belief tables are broadcast over all other modes.
           num_in  : The number of input elements that should be found, i.e. num_in = X.size()[mode_in]
           num_out : The number of output elements to be generated. This must be an integer multiple of the number of logic blocks.   
        '''
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if not num_in % num_arg == 0:
            raise ValueError("The number of input channels must be divisible by the number of logic arguments.")
        num_blk = num_in // num_arg
        if not num_out % num_blk == 0:
            raise ValueError("The number of output channels must be divisible by the number of logic blocks (num_in // num_arg).")

        # Store logic format constants.
        self.num_arg = num_arg
        self.mode_in = mode_in
        self.num_in = num_in
        self.num_out = num_out
        self.b_out = self.num_out // num_blk

        # Construct Hadamard matrix for belief table.
        H = torch.tensor(1., **factory_kwargs).reshape((1, 1))
        for i in range(num_arg):
            H = torch.cat((torch.cat((H, H), 0), torch.cat((-H, H), 0)), 1)
        self.H = H

        # Create belief table parameters, <2**num_arg, num_out>.
        self.P = torch.nn.Parameter(torch.empty((2**self.num_arg, self.num_out), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.P)

    def forward(self, X: Tensor) -> Tensor:
        ''' Input:
           X : Tensor ( * , num_in (mode_in), * ), i.e. the mode specified by mode_in has num_in elements.
        
        Output:
           Y : Tensor ( * , num_out (mode_in), * )
        '''
        # Save size of input X for reshaping output Y.
        X_size = torch.tensor(X.size(), requires_grad=False)
        if X_size[self.mode_in] != self.num_in:
            raise ValueError('Input mode {} contains {} rather than {} elements.'.format(
                              self.mode_in, X_size[self.mode_in], self.num_in))
        Y_size = X_size.detach().clone()
        Y_size[self.mode_in] = self.num_out
        num_slow = torch.prod(X_size[:self.mode_in])
        num_fast = 1 if len(X_size)==self.mode_in+1 else torch.prod(X_size[self.mode_in+1:])
 
        # Form belief table from parameters using Hadamard basis.
        if self.H.device != self.P.device or self.H.dtype != self.P.dtype:
            self.H = self.H.to(device=self.P.device, dtype=self.P.dtype)
        B = self.H.matmul(self.P).view((1, 2**self.num_arg, self.num_out, 1))
 
        Y = LogicFunction.apply(X.view((num_slow, self.num_in, num_fast)),
                                B.view((1, 2**self.num_arg, self.num_out, 1)),
                                self.num_arg, self.b_out)
        return Y.view(list(Y_size))

    def extra_repr(self) -> str:
        return 'Logic 1.1: num_arg={}, mode_in={}, num_in={}, b_out={}, num_out={}'.format(
                self.num_arg, self.mode_in, self.num_in, self.b_out, self.num_out)

