# coding: utf-8
from common.np import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x: np.ndarray):
    x -= x.max(axis=-1, keepdims=True)
    x = np.exp(x)
    x /= x.sum(axis=-1, keepdims=True)
    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    # if t.size == y.size:
    #     t = t.argmax(axis=1)
             
    size = y.shape[0]

    return -np.sum(np.log(y + 1e-7) * t) / size

def normalize(x: np.ndarray):
    mu = np.mean(x)
    sigma = np.sqrt(np.mean((x - mu) ** 2))
    return (x - mu) / sigma