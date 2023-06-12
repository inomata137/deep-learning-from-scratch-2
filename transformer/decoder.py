from ffn import PositionWiseFfn
from residual_connection import ResidualConnection
from attention import MultiheadCrossAttention, MultiheadSelfAttention
from pe import pe
import sys
sys.path.append('..')
from common.np import *  # import numpy as np

rn = np.random.randn

class Decoder:
    def __init__(self, d_m, h, d_ff, repeat_num=6, rn=rn) -> None:
        assert d_m % h == 0
        W1_shape = d_m, d_ff
        b1_shape = 1, d_ff
        W2_shape = d_ff, d_m
        b2_shape = 1, d_m
        self.layers = [[
            ResidualConnection(MultiheadSelfAttention(d_m, h, True, rn)),
            ResidualConnection(MultiheadCrossAttention(d_m, h, rn)),
            ResidualConnection(PositionWiseFfn(
                rn(*W1_shape) / np.sqrt(d_m),
                rn(*b1_shape) * 0.1,
                rn(*W2_shape) / np.sqrt(d_ff),
                rn(*b2_shape) * 0.1
            ))
        ] for _ in range(repeat_num)]
        self.params = []
        self.grads = []
        for layer in self.layers:
            for sublayer in layer:
                self.params += sublayer.params
                self.grads += sublayer.grads
        
    def forward(self, x, hs, train_flg=True, epoch=0):
        for layer in self.layers:
            sa, at, pf = layer
            # from matplotlib import pyplot as plt
            # plt.subplot(221, title='input').imshow(x[0])
            x = sa.forward(x, train_flg=train_flg, epoch=epoch)
            # plt.subplot(222, title='self attention').imshow(x[0])
            x = at.forward(x, hs, train_flg=train_flg, epoch=epoch)
            # plt.subplot(223, title='cross-attention').imshow(x[0])
            x = pf.forward(x, train_flg=train_flg, epoch=epoch)
            # plt.subplot(224, title='ffn').imshow(x[0])
            # match input():
            #     case 's':
            #         plt.show()
            #     case 'q':
            #         exit()
            #     case _:
            #         continue
        return x
    
    def backward(self, dx):
        dhs = None
        for layer in reversed(self.layers):
            sa, at, pf = layer
            dx = pf.backward(dx)
            dx, _dhs = at.backward(dx)
            dx = sa.backward(dx)
            if dhs is None:
                dhs = _dhs
            else:
                dhs += _dhs
        return dx, dhs