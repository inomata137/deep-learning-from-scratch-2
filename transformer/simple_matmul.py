# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np

class SimpleMatMul:
    def __init__(self):
        self.params, self.grads = [], []
        self.a, self.b = None, None
        self.H = None

    def forward(self, a, b):
        self.a, self.b = a, b
        return np.matmul(a, b)

    def backward(self, dout):
        da = np.matmul(dout, self.b.transpose((0, 2, 1)))
        db = np.matmul(self.a.transpose((0, 2, 1)), dout)
        return da, db