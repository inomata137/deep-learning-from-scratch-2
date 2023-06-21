from ffn import PositionWiseFfn
from common.np import *  # import numpy as np
rn = np.random.randn

batch = 3
n = 29
d_m = 64
d_ff = 256

layer = PositionWiseFfn(d_m, d_ff, 0.1, 0.1, rn)

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