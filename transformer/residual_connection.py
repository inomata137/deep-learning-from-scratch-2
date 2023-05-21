import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from base import Layer

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
        n = np.size(x)
        mu = np.sum(x) / n
        sqsum = np.sum((x - mu) ** 2 / n)
        sigma = np.sqrt(sqsum) + self.eps
        self.mu = mu
        self.sigma = sigma
        self.x = x.reshape((1, n))
        o = (x - mu) / sigma
        return o

    def backward(self, dout):
        mu = self.mu
        sigma = self.sigma
        x = self.x
        n = np.size(dout)
        shape = np.shape(dout)
        dout = np.reshape(dout, (1, n))
        j1 = (np.identity(n) * n - 1) / (n * sigma)
        j2 = np.dot(x.T - mu, x - mu) / (n * sigma**3)
        j = j1 - j2
        dx = np.dot(dout, j)
        return np.reshape(dx, shape)

class ResidualConnection:
    def __init__(self, layer: Layer):
        self.layer = layer
        self.ln = LayerNorm()
        self.params = layer.params
        self.grads = layer.grads
    
    def forward(self, x):
        s = x + self.layer.forward(x)
        out = self.ln.forward(s)
        return out

    def backward(self, dout):
        dx = self.ln.backward(dout)
        dx += self.layer.backward(dx)
        return dx
