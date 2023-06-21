from encoder import Encoder
from decoder import Decoder
from pe import pe
import pickle
import sys
sys.path.append('..')
from common.np import *
from common.layers import SoftmaxWithLoss, MatMul, Embedding, Dropout
from common.base_model import BaseModel

rn = np.random.randn

class Transformer(BaseModel):
    def __init__(self, d_m: int, h: int, d_ff: int, vocab_size: int, enc_rep: int, dec_rep: int, p_drop_embed: float, p_drop_sublayer: float, rn=rn):
        self.embed = Embedding(rn(vocab_size, d_m))
        self.dropout_enc = Dropout(p_drop_embed)
        self.dropout_dec = Dropout(p_drop_embed)
        self.enc = Encoder(d_m, h, d_ff, enc_rep, p_drop_sublayer, rn)
        self.dec = Decoder(d_m, h, d_ff, dec_rep, p_drop_sublayer, rn)
        self.matmul = MatMul(rn(d_m, vocab_size) / np.sqrt(d_m))
        self.softmax = SoftmaxWithLoss(e_ls=0.01)

        self.vocab_size = vocab_size

        self.params = self.embed.params + self.enc.params + self.dec.params + self.matmul.params
        self.grads = self.embed.grads + self.enc.grads + self.dec.grads + self.matmul.grads
    
    def forward(self, x_enc, x_dec, epoch=0):
        '''
        x_enc: N x n
        x_dec: N x m
        '''
        vs = self.vocab_size
        N, n = x_enc.shape
        _, m = x_dec.shape
        self.N = N
        self.m = m
        x_embedded = self.embed.forward(np.hstack((x_enc, x_dec)))
        x_enc_embedded = x_embedded[:, :n, :]
        x_dec_embedded = x_embedded[:, n:, :]
        x_enc_embedded += pe(x_enc_embedded)
        x_dec_embedded += pe(x_dec_embedded)
        x_enc_embedded = self.dropout_enc.forward(x_enc_embedded)
        x_dec_embedded = self.dropout_dec.forward(x_dec_embedded)
        hs = self.enc.forward(x_enc_embedded, epoch=epoch)
        y = self.dec.forward(x_dec_embedded, hs, epoch=epoch)
        y = self.matmul.forward(y)
        loss = self.softmax.forward(
            y.reshape((N * m, vs)),
            np.roll(np.eye(vs)[x_dec], -1, 1).reshape((N * m, vs))
        )
        correct_count = (
            y.argmax(-1) == np.roll(np.asarray(x_dec), -1, 1)
        ).all(axis=-1).sum()
        return loss, correct_count.item()
    
    def backward(self, dout=None):
        N = self.N
        m = self.m
        vs = self.vocab_size
        dout = self.softmax.backward().reshape((N, m, vs))
        dout = self.matmul.backward(dout)
        dx_dec, dhs = self.dec.backward(dout)
        dx_enc = self.enc.backward(dhs)
        dx_enc = self.dropout_enc.backward(dx_enc)
        dx_dec = self.dropout_dec.backward(dx_dec)
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
            x_enc_encoded = self.dropout_enc.forward(x_enc_encoded, False)
            x_dec_encoded = self.dropout_dec.forward(x_dec_encoded, False)
            hs = self.enc.forward(x_enc_encoded, False)
            y = self.dec.forward(x_dec_encoded, hs, False)
            y = self.matmul.forward(y)
            x_dec[0, i + 1] = np.argmax(y[0, i])

        return x_dec[0]
