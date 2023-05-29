# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import MatMul, Softmax
from simple_matmul import SimpleMatMul

class SelfAttention:
    def __init__(self, Wi, Wo, mask=False):
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
        self.mask = mask
        self.params = self.input_matmul.params + self.output_matmul.params
        self.grads = self.input_matmul.grads + self.output_matmul.grads
    
    def forward(self, x):
        d_k = self.d_k
        xw = self.input_matmul.forward(x)
        q = xw[:, :, :d_k]
        k = xw[:, :, d_k:2*d_k]
        v = xw[:, :, 2*d_k:]
        a = self.simple_matmul1.forward(q, k.swapaxes(-2, -1)) / np.sqrt(d_k)
        if self.mask:
            a = mask_forward(a)
        sm = self.softmax_layer.forward(a)
        att = self.simple_matmul2.forward(sm, v)
        out = self.output_matmul.forward(att)
        return out
    
    def backward(self, dout):
        d_k = self.d_k
        datt = self.output_matmul.backward(dout)
        dsm, dv = self.simple_matmul2.backward(datt)
        da = self.softmax_layer.backward(dsm)
        if self.mask:
            da = mask_backward(da)
        dq, dkT = self.simple_matmul1.backward(da / np.sqrt(d_k))
        dk = dkT.swapaxes(-2, -1)
        dxw = np.dstack((dq, dk, dv))
        dx = self.input_matmul.backward(dxw)
        return dx

def mask_forward(x):
    r, c = x.shape[-2:]
    f = np.zeros((r, c))
    for ri, r in enumerate(f):
        for ci in range(len(r)):
            r[ci] = -np.inf if ri < ci else 0
    return x + f

def mask_backward(dout):
    r, c = dout.shape[-2:]
    return dout * np.tri(r, c, 0)
