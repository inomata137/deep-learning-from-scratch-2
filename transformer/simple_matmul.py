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
        return np.dot(a, b)

    def backward(self, dout):
        da = np.dot(dout, self.b.T)
        db = np.dot(self.a.T, dout)
        return da, db