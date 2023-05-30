from encoder import Encoder
from decoder import Decoder
from simple_matmul import SimpleMatMul
from pe import pe
import pickle
import sys
sys.path.append('..')
from common.np import *
from common.layers import SoftmaxWithLoss
from common.base_model import BaseModel

np.random.seed(2023)
rn = np.random.randn
with open('./embed.pkl', 'rb') as f:
    embed_weight = pickle.load(f)

class Transformer:
    def __init__(self, d_m, d_k, d_v, d_ff, enc_rep, dec_rep, rn=rn):
        self.enc = Encoder(d_m, d_k, d_v, d_ff, enc_rep, rn)
        self.dec = Decoder(d_m, d_k, d_v, d_ff, dec_rep, rn)
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

class TransformerSeq2Seq(BaseModel):
    def __init__(self, vocab_size, d_m, d_k, d_v, d_ff, enc_rep, dec_rep):
        self.W = embed_weight
        args = d_m, d_k, d_v, d_ff, enc_rep, dec_rep, rn
        self.layer = Transformer(*args)
        self.vocab_size = vocab_size
        self.input_matmul = SimpleMatMul()
        self.output_matmul = SimpleMatMul()
        self.softmax = SoftmaxWithLoss()
        self.params = self.layer.params
        self.grads = self.layer.grads
    
    def forward(self, x_enc, x_dec):
        '''
        x_enc: N x n
        x_dec: N x m
        '''
        W = self.W
        N, n = x_enc.shape
        _, m = x_dec.shape
        vs = self.vocab_size
        self.N = N
        self.m = m
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
        loss = self.softmax.forward(out.reshape((N * m, vs)), np.roll(x_dec_one_hot, -1, -1).reshape((N * m, vs)))

        return loss
    
    def backward(self, dout=1.):
        N = self.N
        m = self.m
        vs = self.vocab_size
        dout = self.softmax.backward().reshape((N, m, vs))
        dout, dwt = self.output_matmul.backward(dout)
        dx_dec, dx_enc = self.layer.backward(dout)
        dx = np.hstack((dx_enc, dx_dec))
        dx, dw = self.input_matmul.backward(dx)
        # dw += dwt.swapaxes(-2, -1)
        # dw = dw.sum(axis=0)
        return dx

    def generate(self, x_enc, start_id, length):
        '''
        x_enc: 1 x n
        '''
        _, n = x_enc.shape
        W = self.W
        x_dec = np.zeros((1, length), dtype=int)
        x_dec[0, 0] = start_id
        vs = self.vocab_size
        x_enc_one_hot = np.eye(vs)[x_enc]
        # N x n x vs

        for i in range(length):
            x_dec_one_hot = np.eye(vs)[x_dec]
            # N x m x vs
            x_one_hot = np.hstack((x_enc_one_hot, x_dec_one_hot))
            # N x (n+m) x vs
            x_encoded = self.input_matmul.forward(x_one_hot, W)
            x_enc_encoded = x_encoded[:, :n, :]
            x_dec_encoded = x_encoded[:, n:, :]
            out = self.layer.forward(x_enc_encoded, x_dec_encoded)
            out = self.output_matmul.forward(out, W.T)[0, i]
            # vs-size one-hot-like vector
            if i + 1 < length:
                x_dec[0, i + 1] = np.argmax(out)
        
        return x_dec[0]
    