# coding: utf-8
from transformer import Transformer
import sys
sys.path.append('..')
# import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from common.np import *

# データの読み込み
(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 訓練データを減らす
# x_train = x_train[:1000]
# t_train = t_train[:1000]
x_test = x_train[:100]
t_test = t_train[:100]

# ハイパーパラメータの設定
vocab_size = len(char_to_id)
d_m = wordvec_size = 32
d_ff = hidden_size = 8
h = 2
enc_rep = 1
dec_rep = 1
batch_size = 128
max_epoch = 30
max_grad = 10.0
p_drop_embed = 0.05
p_drop_sublayer = 0.25
seed = 2023
lr = 0.002

print('-' * 10)
print(f'd_m = {d_m}')
print(f'd_ff = {d_ff}')
print(f'h = {h}')
print(f'enc_rep = {enc_rep}')
print(f'dec_rep = {dec_rep}')
print(f'batch_size = {batch_size}')
print(f'max_grad = {max_grad}')
print(f'p_drop_embed = {p_drop_embed}')
print(f'p_drop_sublayer = {p_drop_sublayer}')
print(f'seed = {seed}')
print(f'lr = {lr}')
print('-' * 10)

np.random.seed(seed)

model = Transformer(d_m, h, d_ff, vocab_size, enc_rep, dec_rep, p_drop_embed, p_drop_sublayer, np.random.randn)
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)

optimizer = Adam(lr, beta2=0.98)
trainer = Trainer(model, optimizer)

acc_list = []

for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct,
                                    id_to_char, verbose)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))

print(acc_list)

if input('save? ') == 'yes':
    filename = input('filename: ') or None
    model.save_params(filename)
