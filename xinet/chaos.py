def load_array_tf(data, data_arrays, batch_size, is_train=True):
    """构造一个TensorFlow数据迭代器。"""
    dataset = data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset


def load_array_gluon(data, data_arrays, batch_size, is_train=True):
    """构造一个 gluon 数据迭代器。"""
    dataset = data.ArrayDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def load_array_torch(data, data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def import_np(module_name):
    if module_name == 'mxnet':
        from mxnet import np, npx
        npx.set_np()
        np.randn = np.random.randn
        return np
    elif module_name == 'torch':
        import torch as np
        np.array = np.tensor
        np.concatenate = np.cat
        return np
    elif module_name == 'tensorflow':
        from tensorflow.experimental import numpy as np
        return np


def load_array(module_name, data_arrays, batch_size, is_train=True):
    if module_name == 'mxnet':
        from mxnet.gluon import data
        return load_array_gluon(data, data_arrays, batch_size, is_train)
    elif module_name == 'torch':
        from torch.utils import data
        return load_array_torch(data, data_arrays, batch_size, is_train)
    elif module_name == 'tensorflow':
        from tensorflow import data
        return load_array_tf(data, data_arrays, batch_size, is_train)
