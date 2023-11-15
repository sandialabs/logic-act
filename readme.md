## Building information for recompiling logic cpp-extension:
Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

Written by Jed A. Duersch, Sandia National Laboratories, Livermore, CA.
Special thanks to Casey Casalnuovo, Sandia National Laboratories, Livermore, CA

This repository implements logical activation functions for PyTorch according to the paper,
"Logical Activation Functions for Training Arbitrary Probabilistic Boolean Operations."

### This build proceedure was completed for a compute node equipped with:
 - NVIDIA A100, CUDA 11.8, Python 3.8, PyTorch 2.0, and GCC 6.5
 - NVIDIA V100, CUDA 12.0, Python 3.8, PyTorch 2.1, and GCC 8.5

### Create a virtual environment with PyTorch
 - python -m my_venv
 - . my_venv/bin/activate
 - pip install torch

### Ensure that the correct virtual environment and compilation utilities are loaded.
 - Depending on your environment, GCC and CUDA may already be available or you may need different versions.
 - module load cuda/11.8
 - module load gcc/6.5

### Then run the setup installation in logic-extension to build.
 - cd logic-extension
 - python setup.py install

