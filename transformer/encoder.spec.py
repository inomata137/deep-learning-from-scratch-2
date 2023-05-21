import sys
sys.path.append('..')
from common.np import np
from common.optimizer import Adam
from common.functions import normalize
from encoder import Encoder
from PIL import Image
from matplotlib import pyplot as plt
import pickle
from pe import pe

input_src = 'images/A.png'
output_src = 'images/todo.jpg'
train_x = normalize(np.array(Image.open(input_src))[:, :, 0])
train_t = normalize(np.array(Image.open(output_src))[:, :, 0])

train_x = train_x.reshape((1, 64, 64))
train_t = train_t.reshape((1, 64, 64))

train_x += pe(train_x)

max_epoch = 100
batch_size = 1
learning_rate = 1e-2
log_interval = 10

model = Encoder(64, 64, 64, 256, 1)
opt = Adam(learning_rate)

loss = []
result = None

for epoch in range(1, max_epoch + 1):
    result = model.forward(train_x)
    diff = result - train_t
    model.backward(diff)
    mse = np.sum(diff ** 2) / 4096
    loss.append(mse)
    opt.update(model.params, model.grads)
    if epoch % log_interval == 0:
        print(f'epoch {epoch}')
        plt.subplot(121, title='output').imshow(result[0])
        plt.subplot(122, title='loss').plot(range(len(loss)), loss)
        plt.show()

# with open('a2b.params.pkl', 'wb') as f:
#     pickle.dump(model.params, f)

fig1 = plt.subplot(121, title=f'final output (epoch={max_epoch})').imshow(result[0], cmap='gray')
fig2 = plt.subplot(122, title='loss')
fig2.set_yscale('log')
fig2.set_ylabel('Mean Square Error')
fig2.set_xlabel('epoch')
fig2.plot(range(1, max_epoch + 1), loss)
fig2.grid()
plt.colorbar(fig1)
plt.show()
