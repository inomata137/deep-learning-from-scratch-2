from ffn import PositionWiseFfn
from common.np import *  # import numpy as np
rn = np.random.randn

batch = 3
n = 29
d_m = 64
d_ff = 256

W1 = np.random.randn(d_m, d_ff)
b1 = np.random.randn(1, d_ff)
W2 = np.random.randn(d_ff, d_m)
b2 = np.random.randn(1, d_m)
layer = PositionWiseFfn(W1, b1, W2, b2)

x = np.random.randn(batch, n, d_m)
y = layer.forward(x)
assert y.shape == (batch, n, d_m), 'forward error'

dout = np.random.randn(*y.shape)
# loss := np.sum(y * dout)
grad = layer.backward(dout)
assert grad.shape == x.shape, 'backward error'

# grad test
for _ in range(100):
    dx = rn(*x.shape) * 1e-6
    dy = layer.forward(x + dx) - y
    dloss1 = np.sum(dy * dout)
    dloss2 = np.sum(dx * grad)
    r = dloss1 / dloss2
    assert r > 0.999 and r < 1.001, 'grad error'

print('ok')