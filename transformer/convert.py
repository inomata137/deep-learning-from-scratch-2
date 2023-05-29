import pickle
import numpy

with open('./embed.pkl', 'rb') as f:
    param = pickle.load(f)

if type(param) is not numpy.ndarray:
    exit()

import cupy
with open('./embed.pkl', 'wb') as f:
    pickle.dump(cupy.array(param), f)
