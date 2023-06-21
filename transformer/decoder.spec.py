from decoder import Decoder
from common.np import *

rn = np.random.randn

batch = 3
d_m = 64
h = 8
d_ff = 256
rep = 2
n = 29
m = 31

dec = Decoder(d_m, h, d_ff, rep, 0., rn)

hs = rn(batch, n, d_m)
x = rn(batch, m, d_m)
y = dec.forward(x, hs)
assert y.shape == (batch, m, d_m), 'forward error'

dout = rn(*y.shape)
grad_x, grad_hs = dec.backward(dout)
assert grad_x.shape == x.shape and grad_hs.shape == hs.shape, 'backward error'

for _ in range(100):
    dx = rn(*x.shape) * 1e-10
    dy = dec.forward(x + dx, hs) - y
    dloss1 = np.sum(dy * dout)
    dloss2 = np.sum(dx * grad_x)
    r = dloss1 / dloss2
    assert r > 0.999 and r < 1.001, f'expected r around 1, got {r}'

print('ok')
