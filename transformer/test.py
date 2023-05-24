import numpy as np
rn = np.random.randn

print('a')
a = np.arange(24).reshape((2, 3, 4))
print(a)

b = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])
print('-^-^-^-^')
print(a[b].shape)