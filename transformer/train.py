# coding: utf-8
import sys
sys.path.append('..')
# import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from transformer import Transformer


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

model = Transformer(d_m, h, d_ff, vocab_size, enc_rep, dec_rep)
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)

optimizer = Adam(lr=0.002, beta2=0.98)
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
    model.save_params()

# グラフの描画
# x = np.arange(len(acc_list))
# plt.plot(x, acc_list, marker='o')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.ylim(-0.05, 1.05)
# plt.show()
