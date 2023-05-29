from simple_matmul import SimpleMatMul
import sys
sys.path.append('..')
from common.np import *
rn = np.random.randn

batch = 3
x_r = 29
x_c = 31
w_c = 24

layer = SimpleMatMul()

x = rn(batch, x_r, x_c)
w = rn(batch, x_c, w_c)
y = layer.forward(x, w)
assert y.shape == (batch, x_r, w_c), 'forward error'

dout = rn(*y.shape)
grad_x, grad_w = layer.backward(dout)
assert grad_x.shape == x.shape
assert grad_w.shape == (batch, x_c, w_c)

for _ in range(100):
    dx = rn(*x.shape) * 1e-6
    dy = layer.forward(x + dx, w) - y
    dloss1 = np.sum(dy * dout)
    dloss2 = np.sum(dx * grad_x)
    r = dloss1 / dloss2
    assert r > 0.999 and r < 1.001, 'grad error'

for _ in range(100):
    dw = rn(*w.shape) * 1e-6
    dy = layer.forward(x, w + dw) - y
    dloss1 = np.sum(dy * dout)
    dloss2 = np.sum(dw * grad_w)
    r = dloss1 / dloss2
    assert r > 0.999 and r < 1.001, 'grad error'

print('ok')