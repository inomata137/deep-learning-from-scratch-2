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
    def __init__(self, W1, b1, W2, b2):
        '''
        W1: d_m x d_ff
        b1: 1 x d_ff
        W2: d_ff x d_m
        b2: 1 x d_m
        '''
        self.relu_layer = Relu()
        self.affine_layers = [
            Affine(W1, b1),
            Affine(W2, b2)
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
