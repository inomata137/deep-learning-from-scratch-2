import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from ffn import PositionWiseFfn
from residual_connection import ResidualConnection
from self_attention import SelfAttention
from positional_encoder import pe

rn = np.random.randn

class AttentionEncoder:
    def __init__(self, d_m, d_k, d_v, d_ff, repeat_num=6):
        Wi_shape = d_m, 2 * d_k + d_v
        Wo_shape = d_v, d_m
        W1_shape = d_m, d_ff
        b1_shape = 1, d_ff
        W2_shape = d_ff, d_m
        b2_shape = 1, d_m
        params: list[list[list]] = [
            [
                [
                    rn(*Wi_shape),
                    rn(*Wo_shape)
                ],
                [
                    rn(*W1_shape) / np.sqrt(d_m),
                    rn(*b1_shape),
                    rn(*W2_shape) / np.sqrt(d_ff),
                    rn(*b2_shape),
                ]
            ] for _ in range(repeat_num)
        ]
        grads = [
            [
                [
                    np.empty_like(param) for param in sublayer_params
                ] for sublayer_params in layer_params
            ] for layer_params in params
        ]
        self.layers = [
            [
                ResidualConnection(SelfAttention(*layer_param[0])),
                ResidualConnection(PositionWiseFfn(*layer_param[1]))
            ] for layer_param in params
        ]
        self.params = sum(sum(params, []), [])
        self.grads = sum(sum(grads, []), [])
    
    def forward(self, x):
        for layer in self.layers:
            for sublayer in layer:
                x = sublayer.forward(x)
        return x
    
    def backward(self, dx):
        for layer in reversed(self.layers):
            for sublayer in reversed(layer):
                dx = sublayer.backward(dx)
        return dx
    
if __name__ == '__main__':
    words_len = 64
    d_model = 64
    h = 1
    d_k = int(d_model / h)
    d_v = d_k
    d_ff = 64
    enc = AttentionEncoder(d_model, d_k, d_v, d_ff, repeat_num=2)
    i = np.random.randn(words_len, d_model)
    i += pe(i)
    o = enc.forward(i)
    do = np.random.randn(*np.shape(o)) * 0.001
    di = enc.backward(do)
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(221, title='input')
    ax2 = fig.add_subplot(222, title='output')
    ax3 = fig.add_subplot(223, title=r'$d_{out}$')
    ax4 = fig.add_subplot(224, title=r'$d_{in}$')
    ax1.imshow(i, cmap='gray')
    ax2.imshow(o, cmap='gray')
    ax3.imshow(do, cmap='gray')
    ax4.imshow(di, cmap='gray')
    fig.savefig('test.png')