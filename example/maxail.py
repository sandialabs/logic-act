''' This code implements MaxAIL activation functions from the paper,
    "Logical Activation Functions: Logit-space equivalents of boolean operators," by S.C. Lowe, R. Earle, J. d'Eon, T. Trappenberg, and S. Oore.
    This implementation was written by Jed A. Duersch, Sandia National Laboratories, Livermore, CA.
'''
import math
import torch
from torch import Tensor

__all__ = ['MaxAIL']

class MaxAIL(torch.nn.Module):
    """ Max-AIL activation functions. """

    __constants__ = ['mode_in', 'num_out', 'b_out', 'num_ch', 'H']
    mode_in: int
    num_in: int
    num_out: int

    def __init__(self,
                 mode_in: int,
                 num_in: int,
                 num_out: int,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if not num_in // 2 == num_out:
            raise ValueError("The number of input channels must be twice the number of output channels.")
        self.mode_in = mode_in
        self.num_in = num_in
        self.num_out = num_out

    def forward(self, X: Tensor) -> Tensor:
        input_size = torch.tensor(X.size(), requires_grad=False)
        output_size = input_size.detach().clone()
        output_size[self.mode_in] = self.num_out
        num_slow = torch.prod(input_size[:self.mode_in])
        num_fast = torch.prod(input_size[self.mode_in+1:])
        X = X.view((num_slow, self.num_out, 2, num_fast))
        Y = torch.maximum(torch.maximum(X[:,:,0,:], X[:,:,1,:]), X[:,:,0,:]+X[:,:,1,:]).view(list(output_size))
        return Y

