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
input1 = rn(batch, words1_len, d_m)
input2 = rn(batch, words2_len, d_m)

layer = Attention(Wi, Wo)

# forward test
output = layer.forward(input2, input1)
assert output.shape == (batch, words2_len, d_m), 'forward error'

# backward test
# loss := np.sum(output * dout)
dout = np.random.randn(*output.shape)
grad2, grad1 = layer.backward(dout)
assert grad1.shape == input1.shape and grad2.shape == input2.shape, 'backward error'
