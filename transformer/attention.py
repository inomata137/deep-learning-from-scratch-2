# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import MatMul, Softmax
from simple_matmul import SimpleMatMul

class Attention:
    def __init__(self, Wi, Wo):
        '''
        Wi: d_m x (2d_k + d_v)
        Wo: d_v x d_m
        '''
        d_v, d_m = np.shape(Wo)
        d_k = int((np.shape(Wi)[1] - d_v) / 2)
        self.matmul_q = SimpleMatMul()
        self.matmul_kv = SimpleMatMul()
        self.matmul_qk = SimpleMatMul()
        self.matmul_at = SimpleMatMul()
        self.matmul_o = MatMul(Wo)
        self.softmax_layer = Softmax()
        self.d_k = d_k
        self.params = [Wi, self.matmul_o.params[0]]
        self.grads = [np.zeros_like(Wi), self.matmul_o.grads[0]]
    
    def forward(self, x, hs):
        d_k = self.d_k
        Wi, Wo = self.params
        kv = self.matmul_kv.forward(hs, Wi[:, d_k:])
        k = kv[:, :, :d_k]
        v = kv[:, :, d_k:]
        q = self.matmul_q.forward(x, Wi[:, :d_k])
        a = self.matmul_qk.forward(q, k.swapaxes(-2, -1)) / np.sqrt(d_k)
        sm = self.softmax_layer.forward(a)
        att = self.matmul_at.forward(sm, v)
        out = self.matmul_o.forward(att)
        return out
    
    def backward(self, dout):
        d_k = self.d_k
        datt = self.matmul_o.backward(dout)
        dsm, dv = self.matmul_at.backward(datt)
        da = self.softmax_layer.backward(dsm) * np.sqrt(d_k)
        dq, dkt = self.matmul_qk.backward(da)
        dk = dkt.swapaxes(-2, -1)
        dq, dwq = self.matmul_q.backward(dq)
        dkv, dwkv = self.matmul_kv.backward(np.dstack((dk, dv)))
        dwi = np.hstack((dwq.sum(axis=0), dwkv.sum(axis=0)))
        self.grads[0][...] = dwi
        return dq, dkv

