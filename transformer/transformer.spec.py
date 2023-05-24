from transformer import TransformerSeq2Seq, Transformer
import sys
sys.path.append('..')
from common.np import *

rn = np.random.randn

vocab_size = 100
d_m = 4
d_k = d_v = d_m
d_ff = 8
enc_rep = 2
dec_rep = 2
batch = 2

i1 = np.array([[
    1, 2, 3, 4, 5
]])
i2 = np.array([[
    10, 3, 5, 1, 4, 2
]])

model = TransformerSeq2Seq(vocab_size, d_m, d_k, d_v, d_ff, enc_rep, dec_rep)
model.forward(i1, i2)
model.backward(np.random.randn(*i2.shape))