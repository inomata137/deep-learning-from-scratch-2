from encoder import Encoder
from decoder import Decoder
from simple_matmul import SimpleMatMul
from pe import pe
import sys
sys.path.append('..')
from common.np import *
from common.layers import SoftmaxWithLoss

np.random.seed(2023)
rn = np.random.randn

class Transformer:
    def __init__(self, d_m, d_k, d_v, d_ff, enc_rep, dec_rep):
        self.enc = Encoder(d_m, d_k, d_v, d_ff, enc_rep)
        self.dec = Decoder(d_m, d_k, d_v, d_ff, dec_rep)
        self.params = self.enc.params + self.dec.params
        self.grads = self.enc.grads + self.dec.grads
    
    def forward(self, i, o):
        '''
        i: N x n x d_m
        o: N x m x d_m
        '''
        i += pe(i)
        o += pe(o)
        hs = self.enc.forward(i)
        o = self.dec.forward(o, hs)
        return o
    
    def backward(self, dout=1.):
        dx_dec, (dhs,) = self.dec.backward(dout)
        dx_enc = self.enc.backward(dhs)
        return dx_dec, dx_enc

class TransformerSeq2Seq:
    def __init__(self, vocab_size, d_m, d_k, d_v, d_ff, enc_rep, dec_rep):
        W = np.random.randn(vocab_size, d_m)
        args = d_m, d_k, d_v, d_ff, enc_rep, dec_rep
        self.layer = Transformer(*args)
        self.vocab_size = vocab_size
        self.input_matmul = SimpleMatMul()
        self.output_matmul = SimpleMatMul()
        self.softmax = SoftmaxWithLoss()
        self.params = [W] + self.layer.params
        self.grads = [np.zeros_like(W)] + self.layer.grads
    
    def forward(self, x_enc, x_dec):
        '''
        x_enc: N x n
        x_dec: N x m
        '''
        W = self.params[0]
        N, n = x_enc.shape
        _, m = x_dec.shape
        vs = self.vocab_size
        x_enc_one_hot = np.eye(vs)[x_enc]
        # N x n x vs
        x_dec_one_hot = np.eye(vs)[x_dec]
        # N x m x vs
        x_one_hot = np.hstack((x_enc_one_hot, x_dec_one_hot))
        # N x (n+m) x vs
        x_encoded = self.input_matmul.forward(x_one_hot, W)
        x_enc_encoded = x_encoded[:, :n, :]
        x_dec_encoded = x_encoded[:, n:, :]

        out = self.layer.forward(x_enc_encoded, x_dec_encoded)
        out = self.output_matmul.forward(out, W.T)
        loss = self.softmax.forward(out, np.roll(x_dec_one_hot, -1, -1))
        return loss
    
    def backward(self, dout=1.):
        dout = self.softmax.backward()
        dout, dwt = self.output_matmul.backward(dout)
        dx_dec, dx_enc = self.layer.backward(dout)
        dx = np.hstack((dx_enc, dx_dec))
        _, dw = self.input_matmul.backward(dx)
        dw += dwt.swapaxes(-2, -1)
        dw = dw.sum(axis=0)
        self.grads[0][...] = dw
