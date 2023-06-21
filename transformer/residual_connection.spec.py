from base import Layer
from residual_connection import ResidualConnection, LayerNorm
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import MatMul
rn = np.random.randn

# LayerNorm test
batch = 3
n = 29
d_m = 32
layer = LayerNorm()

x = rn(batch, n, d_m)
y = layer.forward(x)
assert y.shape == x.shape

dout = rn(*y.shape)
grad = layer.backward(dout)
assert grad.shape == x.shape

for _ in range(100):
    dx = rn(*x.shape) * 1e-6
    dy = layer.forward(x + dx) - y
    dloss1 = np.sum(dy * dout)
    dloss2 = np.sum(dx * grad)
    r = dloss1 / dloss2
    assert r > 0.999 and r < 1.001

# ResidualConection test
batch = 3
n = 29
d_m = 32

W = rn(d_m, d_m)
layer = ResidualConnection(MatMul(W), 0.)

x = rn(batch, n, d_m)
y = layer.forward(x)
assert y.shape == (batch, n, d_m)

dout = rn(*y.shape)
grad = layer.backward(dout)
assert grad.shape == x.shape

for _ in range(100):
    dx = rn(*x.shape) * 1e-7
    dy = layer.forward(x + dx) - y
    dloss1 = np.sum(dy * dout)
    dloss2 = np.sum(dx * grad)
    r = dloss1 / dloss2
    assert r > 0.999 and r < 1.001, f'expected r around 1, got {r}'

print('ok')