from common.np import *
from matplotlib import pyplot as plt
from pe import pe

mat = np.empty((2, 100, 512))
plt.imshow(pe(mat)[0], cmap='gray')
plt.show()
