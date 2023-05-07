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
        n = np.size(x)
        mu = np.sum(x) / n
        shape = np.shape(x)
        x = np.reshape(x, (1, n))
        sqsum = np.sum((x - mu) * (x - mu))
        sigma = np.sqrt(sqsum / n) + self.eps
        self.mu = mu
        self.sigma = sigma
        self.x = x
        o = (x - mu) / sigma
        return np.reshape(o, shape)

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
        return self.ln.forward(s)

    def backward(self, dx):
        do = self.ln.backward(dx)
        return do + self.layer.backward(do)
