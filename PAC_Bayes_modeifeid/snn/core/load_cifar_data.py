"""
Standalone CIFAR10 loader (replacement of the copy of the old `keras.datasets.cifar10.load_data` that used to live here).
It only depends on numpy and returns channels-last uint8 arrays: `(x_train, y_train), (x_test, y_test)`.
"""
import os
import pickle

import numpy as np


def load_batch(fpath):
    with open(fpath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d[b'data'].reshape(-1, 3, 32, 32), np.array(d[b'labels'], dtype='uint8')


def load_data(path='CIFAR_data/cifar-10-batches-py'):
    """Loads CIFAR10 dataset from an already extracted `cifar-10-batches-py` directory.

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)
