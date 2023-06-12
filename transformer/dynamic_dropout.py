import sys
sys.path.append('..')
from common.np import *

class DynamicDropout:
    def __init__(self):
        self.params, self.grads = [], []
        self.mask = None

    def forward(self, x: np.ndarray, train_flg=True, epoch=0):
        p = 0.15 * (1 - 1.1 ** epoch)
        if train_flg:
            self.mask = np.random.rand(*x.shape) > p
            return x * self.mask
        else:
            return x * (1.0 - p)

    def backward(self, dout):
        return dout * self.mask