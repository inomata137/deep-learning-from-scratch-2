# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import MatMul, Softmax
from simple_matmul import SimpleMatMul

class SelfAttention:
    def __init__(self, Wi, Wo):
        '''
        Wi: d_m x (2d_k + d_v)
        Wo: d_v x d_m
        '''
        d_v, d_m = np.shape(Wo)
        d_k = int((np.shape(Wi)[1] - d_v) / 2)
        self.input_matmul = MatMul(Wi)
        self.output_matmul = MatMul(Wo)
        self.simple_matmul1 = SimpleMatMul()
        self.simple_matmul2 =  SimpleMatMul()
        self.softmax_layer = Softmax()
        self.d_k = d_k
        self.params = []
        self.grads = []
    
    def forward(self, x):
        d_k = self.d_k
        xw = self.input_matmul.forward(x)
        q = xw[:, :d_k]
        k = xw[:, d_k:2*d_k]
        v = xw[:, 2*d_k:]
        a = self.simple_matmul1.forward(q, k.T) / np.sqrt(d_k)
        sm = self.softmax_layer.forward(a)
        att = self.simple_matmul2.forward(sm, v)
        out = self.output_matmul.forward(att)
        return out
    
    def backward(self, dout):
        d_k = self.d_k
        datt = self.output_matmul.backward(dout)
        dsm, dv = self.simple_matmul2.backward(datt)
        da = self.softmax_layer.backward(dsm)
        dq, dkT = self.simple_matmul1.backward(da * np.sqrt(d_k))
        dk = dkT.T
        dxw = np.hstack((dq, dk, dv))
        dx = self.input_matmul.backward(dxw)
        return dx