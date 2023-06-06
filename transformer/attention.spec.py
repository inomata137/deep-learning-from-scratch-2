from attention import MultiheadCrossAttention, MultiheadSelfAttention
from common.np import *
rn = np.random.randn

batch = 3
n = 29
m = 31
d_m = 64
h = 8

layer = MultiheadCrossAttention(d_m, h, rn)
# forward test
x1 = rn(batch, n, d_m)
x2 = rn(batch, m, d_m)
y = layer.forward(x2, x1)
assert y.shape == (batch, m, d_m)
# backward test
dout = rn(*y.shape)
grad2, grad1 = layer.backward(dout)
assert grad1.shape == x1.shape and grad2.shape == x2.shape
# grad test
for _ in range(100):
    dx1 = rn(*x1.shape) * 1e-8
    dx2 = rn(*x2.shape) * 1e-8
    dy = layer.forward(x2 + dx2, x1 + dx1) - y
    dloss1 = np.sum(dy * dout)
    dloss2 = np.sum(dx1 * grad1) + np.sum(dx2 * grad2)
    r = dloss1 / dloss2
    assert r > 0.999 and r < 1.001, f'r is {r}'
print('ok')

layer = MultiheadSelfAttention(d_m, h, False, rn)
# forward test
x = rn(batch, n, d_m)
y = layer.forward(x)
assert y.shape == (batch, n, d_m)
# backward test
dout = rn(*y.shape)
grad = layer.backward(dout)
assert grad.shape == x.shape
# grad test
for _ in range(100):
    dx = rn(*x.shape) * 1e-8
    dy = layer.forward(x + dx) - y
    dloss1 = np.sum(dy * dout)
    dloss2 = np.sum(dx * grad)
    r = dloss1 / dloss2
    assert r > 0.999 and r < 1.001, f'r is {r}'
print('ok')

layer = MultiheadSelfAttention(d_m, h, True, rn)
# forward test
x = rn(batch, n, d_m)
y = layer.forward(x)
assert y.shape == (batch, n, d_m)
# backward test
dout = rn(*y.shape)
grad = layer.backward(dout)
assert grad.shape == x.shape
# grad test
for _ in range(100):
    dx = rn(*x.shape) * 1e-8
    dy = layer.forward(x + dx) - y
    dloss1 = np.sum(dy * dout)
    dloss2 = np.sum(dx * grad)
    r = dloss1 / dloss2
    assert r > 0.999 and r < 1.001, f'r is {r}'
print('ok')