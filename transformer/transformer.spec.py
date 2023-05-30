from transformer import TransformerSeq2Seq, Transformer
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
d_k = d_v = d_m
d_ff = 24
rep = 2

layer = TransformerSeq2Seq(vs, d_m, d_k, d_v, d_ff, rep, rep)
x_enc = ri(0, vs, (batch, n))
x_dec = ri(0, vs, (batch, m))
y = layer.forward(x_enc, x_dec)
assert type(y) == np.float64

grad = layer.backward()
assert grad.shape == (batch, n+m, vs)
