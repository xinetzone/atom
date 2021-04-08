import sys
from matplotlib import pyplot as plt

from mxnet import gluon
from mxnet import autograd

from .utils import Accumulator, Animator
from .chaos import import_np, load_array as _load_array, load_nn

_name = __name__.split('.')[1]

np = import_np(_name)
nn = load_nn(_name)

xinet = sys.modules[__name__]

# ======================================
## 特定于框架的类，函数等


def get_dataloader_workers():
    """在非Windows的平台上，使用4个进程来读取的数据。"""
    return 0 if sys.platform.startswith('win') else 4


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中。"""
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))


def accuracy(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.astype(y.dtype) == y
    return float(sum(cmp.astype(y.dtype)))


def train_epoch(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）。"""
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # 计算梯度并更新参数
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), len(y))
    # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]


def sgd(params, lr, batch_size):
    """小批量随机梯度下降。"""
    for param in params:
        param[:] = param - lr * param.grad / batch_size


def updater(params, lr, batch_size):
    return sgd(params, lr, batch_size)

# ======================================
## 共同 API


def load_array(data_arrays, batch_size, is_train=True):
    return _load_array(_name, data_arrays, batch_size, is_train)


def normal(x, mu, sigma):
    p = 1 / np.sqrt(2 * np.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)


def one_hot(arr, num_classes):
    return np.eye(num_classes)[arr]


def softmax(X):
    X_exp = np.exp(X)
    partition = np.sum(X_exp, axis=1, keepdims=True)
    return X_exp / partition  # 这里应用了广播机制


def cross_entropy(y_hat, y):
    return -np.log(y_hat[range(len(y_hat)), y])


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), len(y))
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, loss, num_epochs, updater, ylim=None):
    """训练模型（定义见第3章）。"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=ylim,
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    #train_loss, train_acc = train_metrics
