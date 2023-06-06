from encoder import Encoder
import sys
sys.path.append('..')
from common.np import *
rn = np.random.randn

batch = 3
n = 29
d_m = 16
h = 4
d_ff = 24

layer = Encoder(d_m, h, d_ff, 1, rn)

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
    assert r > 0.999 and r < 1.001