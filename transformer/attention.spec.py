from attention import Attention
from common.np import *
rn = np.random.randn

batch = 3
words1_len = 29
words2_len = 31
d_m = 64
h = 1
d_k = int(d_m / h)
d_v = d_k

Wi = rn(d_m, 2 * d_k + d_v) / np.sqrt(d_m)
Wo = rn(d_v, d_m) / np.sqrt(d_v)
layer = Attention(Wi, Wo)

# forward test
x1 = rn(batch, words1_len, d_m)
x2 = rn(batch, words2_len, d_m)
y = layer.forward(x2, x1)
assert y.shape == (batch, words2_len, d_m)

# backward test
# loss := np.sum(output * dout)
dout = np.random.randn(*y.shape)
grad2, grad1 = layer.backward(dout)
assert grad1.shape == x1.shape and grad2.shape == x2.shape

# grad test
for _ in range(100):
    dx1 = rn(*x1.shape) * 1e-7
    dx2 = rn(*x2.shape) * 1e-7
    y2 = layer.forward(x2 + dx2, x1 + dx1)
    dy = y2 - y
    dloss1 = np.sum(dy * dout)
    dloss2 = np.sum(dx1 * grad1) + np.sum(dx2 * grad2)
    r = dloss1 / dloss2
    assert r > 0.999 and r < 1.001

print('ok')
