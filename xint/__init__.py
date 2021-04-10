"""refer to "Dive into Deep Learning" (https://d2l.ai).

Example
================================
from xinet import mxnet as xinet  # Use MXNet as the backend
from xinet import torch as xinet  # Use PyTorch as the backend
from xinet import tensorflow as xinet # Use TensorFlow as the backend
from xinet import utils as xutils

np = xinet.np

def normal(x, mu, sigma):
    p = 1 / np.sqrt(2 * np.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
================================
"""

__version__ = "0.0.1"
