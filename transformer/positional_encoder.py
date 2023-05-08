import sys
sys.path.append('..')
from common.np import *

def pe(x, scale=1.):
    '''
    x: (n x d_m) matrix
    '''
    r, c = np.shape(x)
    pos = np.array(range(r)).reshape((r, 1))
    dim = np.array(range(c)).reshape((1, c))
    pe_even = np.sin(pos / 10000**(dim / c)) * ((dim + 1) % 2)
    pe_odd = np.cos(pos / 10000**((dim - 1) / c)) * (dim % 2)
    return (pe_even + pe_odd) * scale

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    mat = np.empty((100, 512))
    plt.imshow(pe(mat), cmap='gray')
    plt.show()
