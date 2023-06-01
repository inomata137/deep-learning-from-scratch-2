from encoder import Encoder
from decoder import Decoder
from pe import pe
import pickle
import sys
sys.path.append('..')
from common.np import *
from common.layers import SoftmaxWithLoss, MatMul, Embedding
from common.base_model import BaseModel

np.random.seed(2023)
rn = np.random.randn
# with open('./embed.pkl', 'rb') as f:
#     embed_weight = pickle.load(f)

embed_weight = np.load('./learned_W_embed.npy')

class Transformer(BaseModel):
    def __init__(self, d_m: int, d_k: int, d_v: int, d_ff: int, vocab_size: int, enc_rep=1, dec_rep=1, rn=rn):
        self.embed = Embedding(embed_weight) # (vs, d_m)
        self.enc = Encoder(d_m, d_k, d_v, d_ff, enc_rep, rn)
        self.dec = Decoder(d_m, d_k, d_v, d_ff, dec_rep, rn)
        self.matmul = MatMul(rn(d_m, vocab_size))
        self.softmax = SoftmaxWithLoss()

        self.vocab_size = vocab_size

        self.params = self.embed.params + self.enc.params + self.dec.params + self.matmul.params
        self.grads = self.embed.grads + self.enc.grads + self.dec.grads + self.matmul.grads
    
    def forward(self, x_enc, x_dec):
        '''
        x_enc: N x n
        x_dec: N x m
        '''
        vs = self.vocab_size
        N, n = x_enc.shape
        _, m = x_dec.shape
        self.N = N
        self.m = m
        x_encoded = self.embed.forward(np.hstack((x_enc, x_dec)))
        x_enc_encoded = x_encoded[:, :n, :]
        x_dec_encoded = x_encoded[:, n:, :]
        x_enc_encoded += pe(x_enc_encoded)
        x_dec_encoded += pe(x_dec_encoded)
        hs = self.enc.forward(x_enc_encoded)
        y = self.dec.forward(x_dec_encoded, hs)
        y = self.matmul.forward(y)
        loss = self.softmax.forward(
            y.reshape((N * m, vs)),
            np.roll(np.eye(vs)[x_dec], -1, 1).reshape((N * m, vs))
        )
        return loss
    
    def backward(self, dout=None):
        N = self.N
        m = self.m
        vs = self.vocab_size
        dout = self.softmax.backward().reshape((N, m, vs))
        dout = self.matmul.backward(dout)
        dx_dec, dhs = self.dec.backward(dout)
        dx_enc = self.enc.backward(dhs)
        dx = np.hstack((dx_enc, dx_dec))
        dx = self.embed.backward(dx)
        return dx
    
    def generate(self, x_enc, start_id, length):
        '''
        x_enc: 1 x n
        '''
        vs = self.vocab_size
        _, n = x_enc.shape
        x_dec = np.zeros((1, length), dtype=int)
        x_dec[0, 0] = start_id
        
        for i in range(length - 1):
            x_encoded = self.embed.forward(np.hstack((x_enc, x_dec)))
            x_enc_encoded = x_encoded[:, :n, :]
            x_dec_encoded = x_encoded[:, n:, :]
            x_enc_encoded += pe(x_enc_encoded)
            x_dec_encoded += pe(x_dec_encoded)
            hs = self.enc.forward(x_enc_encoded)
            y = self.dec.forward(x_dec_encoded, hs)
            y = self.matmul.forward(y)
            x_dec[0, i + 1] = np.argmax(y[0, i])

        return x_dec[0]
