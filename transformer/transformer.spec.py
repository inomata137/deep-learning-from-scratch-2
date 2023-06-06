from transformer import Transformer
import sys
sys.path.append('..')
from common.np import *
rn = np.random.randn
ri = np.random.randint

batch = 3
vs = 59
n = 29
m = 31
d_m = 16
h = 4
d_ff = 24
rep = 2

layer = Transformer(d_m, h, d_ff, vs, rep, rep, rn)
x_enc = ri(0, vs, (batch, n))
x_dec = ri(0, vs, (batch, m))
y = layer.forward(x_enc, x_dec)
assert type(y) == np.float64

assert layer.backward()  is None
