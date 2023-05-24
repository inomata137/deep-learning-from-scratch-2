from attention import Attention
from common.np import *

batch = 3
words1_len = 29
words2_len = 31
d_m = 64
h = 1
d_k = int(d_m / h)
d_v = d_k
Wi = np.random.randn(d_m, 2 * d_k + d_v)
Wo = np.random.randn(d_v, d_m)
layer = Attention(Wi, Wo)
input1 = np.random.randn(batch, words1_len, d_m)
input2 = np.random.randn(batch, words2_len, d_m)
output = layer.forward(input2, input1)
assert output.shape == (batch, words2_len, d_m), 'forward error'
dout = np.random.randn(*output.shape)
din2, din1 = layer.backward(dout)
assert din1.shape == input1.shape and din2.shape == input2.shape, 'backward error'

print('ok')
