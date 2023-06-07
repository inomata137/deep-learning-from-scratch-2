import matplotlib.pyplot as plt
from transformer import Transformer
from attention import MultiheadAttention, MultiheadSelfAttention
import sys
sys.path.append('..')
from dataset import sequence
from common.np import *
rn = np.random.randn
np.random.seed(np.random.randint(32768))

(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

d_m = 16
h = 4
d_ff = 16
vs = 59
er = 1
dr = 1

model = Transformer(d_m, h, d_ff, vs, er, dr, rn)
model.load_params('./Transformer.pkl')

model.forward(x_train[:1], t_train[:1])

encoder_attention: MultiheadSelfAttention = model.enc.layers[0][0].layer

def vis(attention: MultiheadAttention):
    for i, head in enumerate(attention.heads):
        weight = head.attention_weight[0]
        plt.subplot(221 + i, title=r'head$_{}$'.format(i+1)).imshow(weight)
    plt.show()

vis(encoder_attention.layer)