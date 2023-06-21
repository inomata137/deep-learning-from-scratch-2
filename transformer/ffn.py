# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Affine, SoftmaxWithLoss
from common.functions import relu
from simple_matmul import SimpleMatMul

class Relu:
    def __init__(self) -> None:
        self.params = []
        self.grads = []
        self.cache = None
    def forward(self, x):
        r = relu(x)
        self.cache = np.heaviside(x, 0.)
        return r
    def backward(self, dx):
        return self.cache * dx

class PositionWiseFfn:
    def __init__(self, d_m: int, d_ff: int, b1_scale=1., b2_scale=1., rn=np.random.randn):
        self.relu_layer = Relu()
        self.affine_layers = [
            Affine(rn(d_m, d_ff) / np.sqrt(d_m), rn(1, d_ff) * b1_scale),
            Affine(rn(d_ff, d_m) / np.sqrt(d_ff), rn(1, d_m) * b2_scale)
        ]
        self.params = []
        self.grads = []
        for layer in self.affine_layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def forward(self, x):
        '''
        x: N x n x d_m
        '''
        t = self.affine_layers[0].forward(x)
        a = self.relu_layer.forward(t)
        o = self.affine_layers[1].forward(a)
        return o
    
    def backward(self, dout):
        da = self.affine_layers[1].backward(dout)
        dt = self.relu_layer.backward(da)
        dx = self.affine_layers[0].backward(dt)
        return dx
