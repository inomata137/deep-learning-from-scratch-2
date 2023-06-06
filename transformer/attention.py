# coding: utf-8
import warnings
# from concurrent import futures
from simple_matmul import SimpleMatMul
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import MatMul, Softmax

warnings.filterwarnings('ignore', 'divide by zero encountered in log')

class AttentionHead:
    def __init__(self, d_m: int, d_k: int, mask: bool, rn=np.random.randn):
        self.wq = MatMul(rn(d_m, d_k))
        self.wk = MatMul(rn(d_m, d_k))
        self.wv = MatMul(rn(d_m, d_k))
        self.matmul1 = SimpleMatMul()
        self.softmax = Softmax()
        self.matmul2 = SimpleMatMul()
        self.params = [
            *self.wq.params,
            *self.wk.params,
            *self.wv.params,
            *self.matmul1.params,
            *self.softmax.params,
            *self.matmul2.params
        ]
        self.grads = [
            *self.wq.grads,
            *self.wk.grads,
            *self.wv.grads,
            *self.matmul1.grads,
            *self.softmax.grads,
            *self.matmul2.grads
        ]
        self.d_k = d_k
        self.mask = mask

    def forward(self, x_q: np.ndarray, x_kv: np.ndarray):
        q = self.wq.forward(x_q)
        k = self.wk.forward(x_kv)
        v = self.wv.forward(x_kv)
        x = self.matmul1.forward(q, k.swapaxes(-2, -1)) / np.sqrt(self.d_k)
        if self.mask:
            r, c = x.shape[-2:]
            x += np.log(np.tri(r, c, 0))
        x = self.softmax.forward(x)
        x = self.matmul2.forward(x, v)
        return x

    def backward(self, dout):
        dx, dv = self.matmul2.backward(dout)
        dx = self.softmax.backward(dx)
        if self.mask:
            r, c = dx.shape[-2:]
            dx *= np.tri(r, c, 0)
        dq, dkT = self.matmul1.backward(dx / np.sqrt(self.d_k))
        dx_q = self.wq.backward(dq)
        dx_k = self.wk.backward(dkT.swapaxes(-2, -1))
        dx_v = self.wv.backward(dv)
        dx_kv = dx_k + dx_v
        return dx_q, dx_kv

class MultiheadAttention:
    def __init__(self, d_m: int, h=1, mask=False, rn=np.random.randn):
        assert d_m % h == 0, 'd_m/h should be an integer'
        d_v = d_k = d_m // h
        self.heads = [AttentionHead(d_m, d_k, mask, rn) for _ in range(h)]
        self.wo = MatMul(rn(d_m, d_m))
        self.params = []
        self.grads = []
        for head in self.heads:
            self.params += head.params
            self.grads += head.grads
        self.params += self.wo.params
        self.grads += self.wo.grads
        self.h = h
        self.d_v = d_v
    
    def forward(self, x_q, x_kv):
        '''
        x: (N, n, d_m)
        '''
        # result: list[futures.Future] = []
        # with futures.ThreadPoolExecutor() as executor:
        #     for head in self.heads:
        #         result.append(executor.submit(head.forward, x_q, x_kv))
        # result = tuple(f.result() for f in result)
        result = tuple(
            head.forward(x_q, x_kv) for head in self.heads
        )
        x = np.dstack(result)
        x = self.wo.forward(x)
        return x
    
    def backward(self, dout):
        '''
        dout: (N, n, d_m)
        '''
        dout = self.wo.backward(dout) # (N, n, h*d_v)
        # result = []
        # with futures.ThreadPoolExecutor() as executor:
        #     for i, head in enumerate(self.heads):
        #         datt = dout[:, :, i*self.d_v:(i+1)*self.d_v]
        #         result.append(executor.submit(head.backward, datt))
        # result = [f.result() for f in result]
        result = [head.backward(
            dout[:, :, i*self.d_v:(i+1)*self.d_v]
        ) for i, head in enumerate(self.heads)]
        dx_q = 0
        dx_kv = 0
        for r in result:
            dx_q += r[0]
            dx_kv += r[1]
        return dx_q, dx_kv

class MultiheadSelfAttention:
    def __init__(self, d_m: int, h=1, mask=False, rn=np.random.randn):
        self.layer = MultiheadAttention(d_m, h, mask, rn)
        self.params = self.layer.params
        self.grads = self.layer.grads
    
    def forward(self, x):
        return self.layer.forward(x, x)

    def backward(self, dout):
        dx_q, dx_kv = self.layer.backward(dout)
        return dx_q + dx_kv

class MultiheadCrossAttention(MultiheadAttention):
    def __init__(self, d_m: int, h=1, rn=np.random.randn):
        super().__init__(d_m, h, False, rn)