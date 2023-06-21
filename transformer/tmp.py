import matplotlib.pyplot as plt
from transformer import Transformer
from attention import MultiheadAttention, MultiheadSelfAttention
import sys
sys.path.append('..')
from dataset import sequence
from common.np import *
rn = np.random.randn
# np.random.seed(np.random.randint(32768))

(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

d_m = 24
h = 2
d_ff = 1
vs = 59
er = 1
dr = 1

model = Transformer(d_m, h, d_ff, vs, er, dr, rn)
model.load_params('./Transformer.pkl')
idx = int(input())
x = x_test[idx:idx+1]
x = np.asarray(x)
out = model.generate(x, 14, 11).flatten()
q = ''.join([id_to_char[i] for i in x.flatten().get()])
a = ''.join([id_to_char[i] for i in out.get()])
print(q)
print(a)

attention = model.dec.layers[0][1].layer

xticks = [*q]
yticks = [*a[1:]] + [' ']
# yticks = [*a]

def vis(attention: MultiheadAttention):
    for i, head in enumerate(attention.heads):
        weight = head.attention_weight[0]
        ax = plt.subplot(211 + i, title=r'head$_{}$'.format(i+1))
        ax.imshow(weight.get())
        ax.set_xticks(range(29))
        ax.set_xticklabels(xticks)
        ax.set_yticks(range(11))
        ax.set_yticklabels(yticks)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.show()

vis(attention)
