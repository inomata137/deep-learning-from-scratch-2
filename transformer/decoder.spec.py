from decoder import Decoder
from common.np import *

rn = np.random.randn

batch = 3
d_m = 64
d_k = 64
d_v = 64
d_ff = 256
rep = 2
n = 29
m = 31

dec = Decoder(d_m, d_k, d_v, d_ff, rep)

hs = rn(batch, n, d_m)
input_dec = rn(batch, m, d_m)
output = dec.forward(input_dec, hs)
assert output.shape == (batch, m, d_m), 'forward error'
dx, dhs = dec.backward(rn(*output.shape))
assert dx.shape == input_dec.shape and dhs[0].shape == hs.shape, 'backward error'
for param in dec.params:
    print(param.shape)
print('ok')
