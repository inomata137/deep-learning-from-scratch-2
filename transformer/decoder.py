from ffn import PositionWiseFfn
from residual_connection import ResidualConnection
from attention import MultiheadCrossAttention, MultiheadSelfAttention
from pe import pe
import sys
sys.path.append('..')
from common.np import *  # import numpy as np

rn = np.random.randn

class Decoder:
    def __init__(self, d_m, h, d_ff, repeat_num: int, p_drop: float, rn=rn) -> None:
        assert d_m % h == 0
        self.layers = [[
            ResidualConnection(MultiheadSelfAttention(d_m, h, True, rn), p_drop),
            ResidualConnection(MultiheadCrossAttention(d_m, h, rn), p_drop),
            ResidualConnection(PositionWiseFfn(d_m, d_ff, 0.1, 0.1, rn), p_drop)
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
            x = sa.forward(x, train_flg=train_flg, epoch=epoch)
            x = at.forward(x, hs, train_flg=train_flg, epoch=epoch)
            x = pf.forward(x, train_flg=train_flg, epoch=epoch)
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
