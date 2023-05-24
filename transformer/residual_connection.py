import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from base import Layer
from attention import Attention

class LayerNorm:
    def __init__(self, eps=1e-10):
        self.mu = None
        self.sigma = None
        self.x = None
        self.eps = eps

    def forward(self, x):
        '''
        x: N x n x d_m tensor
        '''
        N, n, d_m = x.shape
        n *= d_m
        mu = np.sum(x, axis=(1, 2), keepdims=True) / n
        # mu: N x 1 x 1
        sqsum = np.sum((x - mu) ** 2 / n, axis=(1, 2), keepdims=True)
        # sqsum: N x 1 x 1
        sigma = np.sqrt(sqsum) + self.eps
        # sigma: N x 1 x 1
        self.mu = mu
        self.sigma = sigma
        self.x = x
        return (x - mu) / sigma

    def backward(self, dout):
        mu = self.mu
        # mu: N x 1 x 1
        sigma = self.sigma
        # sigma: N x 1 x 1
        x = self.x
        # x: N x n x d_m
        N, n, d_m = dout.shape
        n *= d_m
        j1 = (np.identity(n) * n - 1) / (n * sigma)
        j2 = np.matmul(
            x.reshape((N, n, 1)) - mu,
            x.reshape((N, 1, n)) - mu
        ) / (n * sigma**3)
        j = j1 - j2
        dx = np.matmul(dout.reshape((N, 1, n)), j)
        return np.reshape(dx, dout.shape)

class ResidualConnection:
    def __init__(self, layer: Layer):
        self.layer = layer
        self.ln = LayerNorm()
        self.params = layer.params
        self.grads = layer.grads
    
    def forward(self, x, y=None):
        args = (x,) if y is None else (x, y)
        s = x + self.layer.forward(*args)
        out = self.ln.forward(s)
        return out

    def backward(self, dout):
        dx1 = self.ln.backward(dout)
        dx2 = self.layer.backward(dout)
        if type(dx2) == tuple:
            dx1 += dx2[0]
            return dx1, dx2[1:]
        else:
            return dx1 + dx2
