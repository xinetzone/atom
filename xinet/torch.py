import sys

from .chaos import import_np, load_array as _load_array

_name = __name__.split('.')[1]

np = import_np(_name)
np.pi = np.acos(np.zeros(1)) * 2 

xinet = sys.modules[__name__]

def load_array(data_arrays, batch_size, is_train=True):
    return _load_array(_name, data_arrays, batch_size, is_train)