''' This code implements MaxMin activation functions from the paper,
    "Norm-preserving orthogonal permutation linear unit activation functions" by A. Chernodub and D. Nowicki.
    This implementation was written by Jed A. Duersch, Sandia National Laboratories, Livermore, CA.
'''
import math
import torch
from torch import Tensor

__all__ = ['MinMax']

class MinMax(torch.nn.Module):
    """ Min-max activation functions. """

    __constants__ = ['mode_in', 'num_out', 'b_out', 'num_ch', 'H']
    mode_in: int
    num_ch: int

    def __init__(self,
                 mode_in: int,
                 num_ch: int,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if not num_ch % 2 == 0:
            raise ValueError("The number of input channels must be divisible by 2.")
        self.mode_in = mode_in
        self.num_ch = num_ch
        self.num_blk = self.num_ch // 2

    def forward(self, X: Tensor) -> Tensor:
        input_size = torch.tensor(X.size(), requires_grad=False)
        num_slow = torch.prod(input_size[:self.mode_in])
        num_fast = torch.prod(input_size[self.mode_in+1:])
        X = X.view((num_slow, self.num_blk, 2, num_fast))
        Y = torch.cat((torch.maximum(X[:,:,0,:], X[:,:,1,:]), torch.minimum(X[:,:,0,:], X[:,:,1,:])), 2).view(list(input_size))
        return Y

