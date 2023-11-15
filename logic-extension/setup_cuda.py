"""
Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

Written by Jed A. Duersch, Sandia National Laboratories, Livermore, CA.

This algorithm implements logical activation functions according to the paper,
"Logical Activation Functions for Training Arbitrary Probabilistics Boolean Operations."
"""

from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='logic11_cuda',
      ext_modules=[cpp_extension.CUDAExtension('logic11_cuda', ['logic11_cuda.cpp', 'logic11_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

