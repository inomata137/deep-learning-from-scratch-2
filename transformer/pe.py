import sys
sys.path.append('..')
from common.np import *

def pe(x, scale=1.):
    '''
    x: (n x d_m) matrix
    '''
    b, r, c = np.shape(x)
    pos = np.array(range(r)).reshape((r, 1))
    dim = np.array(range(c)).reshape((1, c))
    pe_even = np.sin(pos / 10000**(dim / c)) * ((dim + 1) % 2)
    pe_odd = np.cos(pos / 10000**((dim - 1) / c)) * (dim % 2)
    _pe = (pe_even + pe_odd) * scale
    _pe = _pe.reshape(1, r, c).repeat(b, axis=b)
    return _pe
