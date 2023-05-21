from ffn import PositionWiseFfn
from common.np import *  # import numpy as np

batch = 3
n = 29
d_m = 64
d_ff = 256
W1 = np.random.randn(d_m, d_ff)
b1 = np.random.randn(1, d_ff)
W2 = np.random.randn(d_ff, d_m)
b2 = np.random.randn(1, d_m)
layer = PositionWiseFfn(W1, b1, W2, b2)

input = np.random.randn(batch, n, d_m)
output = layer.forward(input)
dout = np.random.randn(*output.shape)
din = layer.backward(dout)

from matplotlib import pyplot as plt
plt.subplot(221, title='input').imshow(input[0])
plt.subplot(222, title='output').imshow(output[0])
plt.subplot(223, title='output').imshow(dout[0])
plt.subplot(224, title='output').imshow(din[0])
plt.show()
