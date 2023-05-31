# coding: utf-8
import sys
sys.path.append('..')
# import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from transformer import TransformerSeq2Seq


# データの読み込み
(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 訓練データを減らす
x_train = x_train[:1000]
t_train = t_train[:1000]
x_test = x_test[:12]
t_test = t_test[:12]

# ハイパーパラメータの設定
vocab_size = len(char_to_id)
d_m = wordvec_size = 16
d_ff = hidden_size = 64
d_k = d_v = 16
enc_rep = 1
dec_rep = 1
batch_size = 32
max_epoch = 10
max_grad = 5.0

model = TransformerSeq2Seq(vocab_size, d_m, d_k, d_v, d_ff, enc_rep, dec_rep)
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)

optimizer = Adam(lr=0.01)
trainer = Trainer(model, optimizer)

acc_list = []

correct_num = 0
for i in range(len(x_test)):
    question, correct = x_test[[i]], t_test[[i]]
    verbose = i < 10
    correct_num += eval_seq2seq(model, question, correct,
                                id_to_char, verbose)
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


model.save_params()

print(acc_list)
# グラフの描画
# x = np.arange(len(acc_list))
# plt.plot(x, acc_list, marker='o')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.ylim(-0.05, 1.05)
# plt.show()
