from self_attention import SelfAttention
from common.np import *
rn = np.random.randn

batch = 3
n = 29
d_m = 32
d_k = d_v = d_m

Wi = rn(d_m, 2 * d_k + d_v)
Wo = rn(d_v, d_m)
layer = SelfAttention(Wi, Wo)

# forward test
x = rn(batch, n, d_m)
y= layer.forward(x)
assert y.shape == (batch, n, d_m)

# backward test
dout = rn(*y.shape)
grad = layer.backward(dout)
assert grad.shape == x.shape

# grad test
for _ in range(100):
    dx = rn(*x.shape) * 1e-7
    dy = layer.forward(x + dx) - y
    dloss1 = np.sum(dy * dout)
    dloss2 = np.sum(dx * grad)
    r = dloss1 / dloss2
    assert r > 0.999 and r < 1.001

print('ok')