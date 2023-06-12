from ffn import PositionWiseFfn
from residual_connection import ResidualConnection
from attention import MultiheadSelfAttention
import sys
sys.path.append('..')
from common.np import *  # import numpy as np

rn = np.random.randn

class Encoder:
    def __init__(self, d_m, h, d_ff, repeat_num=6, rn=rn):
        assert d_m % h == 0
        W1_shape = d_m, d_ff
        b1_shape = 1, d_ff
        W2_shape = d_ff, d_m
        b2_shape = 1, d_m
        self.layers = [[
            ResidualConnection(MultiheadSelfAttention(d_m, h, False, rn)),
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
    
    def forward(self, x, train_flg=True, epoch=0):
        for layer in self.layers:
            for sublayer in layer:
                x = sublayer.forward(x, train_flg=train_flg, epoch=epoch)
        return x
    
    def backward(self, dx):
        for layer in reversed(self.layers):
            for sublayer in reversed(layer):
                dx = sublayer.backward(dx)
        return dx
