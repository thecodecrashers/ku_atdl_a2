import gzip
import os
import random
import tarfile
import urllib.request
import pickle

import numpy as np

NUM_CLASSES = 10
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MNIST_DATA_DIR = os.path.join(_PKG_ROOT, "mnist")
CIFAR_DATA_DIR = os.path.join(_PKG_ROOT, "CIFAR_data")
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# The original TensorFlow `input_data.read_data_sets` held out the first 5000 training images as a validation set,
# i.e. the training set had 55000 images. This is reproduced by default.
MNIST_VALIDATION_SIZE = 5000


def binarize_mnist_labels(labels):
    positive_mask = (labels > 4).astype(int).reshape(-1, 1)
    negative_mask = (labels <= 4).astype(int).reshape(-1, 1)
    mask = positive_mask - negative_mask
    print("mask: ", mask)
    return mask


def normalize_meanstd(images, axis=None, mean = None, std = None ):
    # axis param denotes axes along which mean & std reductions are to be performed
    if mean is None:
      mean = np.mean(images, axis=axis, keepdims=True)
      std = np.sqrt(((images - mean) ** 2).mean(axis=axis, keepdims=True))

    out = (images - mean) / std
    return out, mean, std


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)


def _load_raw_mnist(data_dir, validation_size):
    train_images = _read_idx_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    train_labels = _read_idx_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test_images = _read_idx_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    test_labels = _read_idx_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    return ((train_images[validation_size:], train_labels[validation_size:]),
            (test_images, test_labels))


def _one_hot(labels, num_classes=NUM_CLASSES):
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def load_mnist_data(data_dir = MNIST_DATA_DIR, one_hot=True, validation_size=MNIST_VALIDATION_SIZE):
    (trainX, trainY), (testX, testY) = _load_raw_mnist(data_dir, validation_size)
    if one_hot:
        trainY, testY = _one_hot(trainY), _one_hot(testY)
    trainY = np.reshape(trainY.astype(np.float32), [trainX.shape[0], NUM_CLASSES])
    testY = np.reshape(testY.astype(np.float32), [testX.shape[0], NUM_CLASSES])
    return (trainX, trainY), (testX, testY)


def load_binary_mnist(data_dir = MNIST_DATA_DIR, validation_size=MNIST_VALIDATION_SIZE):
    (trainX, trainY), (testX, testY) = _load_raw_mnist(data_dir, validation_size)
    trainY = binarize_mnist_labels(trainY).astype(np.float32)
    testY = binarize_mnist_labels(testY).astype(np.float32)
    return (trainX, trainY), (testX, testY)


def _load_cifar_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    data = d[b"data"].reshape(-1, 3, 32, 32)
    return data, np.array(d[b"labels"], dtype=np.uint8)


def _load_cifar_raw(data_dir):
    """ Loads CIFAR10 as uint8 NHWC arrays, downloading it if it is not present in data_dir. """
    batches_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(os.path.join(batches_dir, "test_batch")):
        os.makedirs(data_dir, exist_ok=True)
        archive = os.path.join(data_dir, "cifar-10-python.tar.gz")
        if not os.path.exists(archive):
            print("Downloading CIFAR10 from", CIFAR_URL)
            urllib.request.urlretrieve(CIFAR_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(data_dir)
    xs, ys = zip(*[_load_cifar_batch(os.path.join(batches_dir, "data_batch_%d" % i)) for i in range(1, 6)])
    x_train, y_train = np.concatenate(xs).transpose(0, 2, 3, 1), np.concatenate(ys).reshape(-1, 1)
    x_test, y_test = _load_cifar_batch(os.path.join(batches_dir, "test_batch"))
    return (x_train, y_train), (x_test.transpose(0, 2, 3, 1), y_test.reshape(-1, 1))


def load_cifar_data(data_dir = CIFAR_DATA_DIR, one_hot=True):
    """ Loads CIFAR10 data and whitens it (per-pixel mean/std computed on the training set).
    :param data_dir: directory where cifar data is stored (downloaded there if missing).
    :param one_hot: BOOL, if True do one-hot encoding of the labels
    :return: returns (images, labels) for train and test data
    """

    (x_train, y_train), (x_test, y_test) = _load_cifar_raw(data_dir)
    y_train = _one_hot(y_train[:, 0], NUM_CLASSES)
    y_test = _one_hot(y_test[:, 0], NUM_CLASSES)
    x_train, xmean, xstd = normalize_meanstd(x_train, axis=0)
    x_test,_m,_s = normalize_meanstd(x_test, axis=0, mean=xmean, std=xstd)
    return (x_train.astype(np.float32), y_train), (x_test.astype(np.float32), y_test)


def next_batch(xx,yy, batchsize, idx):
    return [xx[idx*batchsize:(idx+1)*batchsize], yy[idx*batchsize:(idx+1)*batchsize]]


def shuffledata(xx,yy):
    ns = xx.shape[0]
    idx = random.sample(list(np.arange(ns)), ns)
    xx = xx[idx]
    yy = yy[idx]
    return xx,yy


def sort_corresponding_to_labels(trainX, trainY, label_order):
    """ Sorts the x and y such that they match the training order in label order. Tested. """
    trainY_labels = np.where(trainY > 0)[1]

    idxs = np.arange(0, len(trainY_labels))
    idmap = dict((id,pos) for pos,id in enumerate(label_order))
    l_ordered = [(x, y, _, _o) for _, _o, x, y in sorted(zip(idxs, trainY_labels, trainX, trainY), key=lambda x:idmap[x[1]])]
    newlabord = [l_ordered[i][3] for i in range(len(l_ordered))]
    x_ordered = np.array([l_ordered[i][0] for i in range(len(l_ordered))])
    y_ordered = np.array([l_ordered[i][1] for i in range(len(l_ordered))])

    return x_ordered, y_ordered, newlabord


def concat_labels(label_order):
    return np.hstack(label_order)


def label_indices(x, L):
    """ Returns the indices of data points with label L. TESTED."""
    labels = np.where(x > 0)[1]
    return np.where(labels == L)[0]


def sample_label(x, L):
    """ Samples a random instance with label L. Returns the index of the data point location. """
    idx = label_indices(x, L)
    return np.random.choice(idx)


def sample_label_from_dict(dd, L, size=None, replace=True):
    if isinstance(L, np.ndarray):
        return np.random.choice(dd[L[0]], size=size, replace=replace)
    else:
        return np.random.choice(dd[L], size=size, replace=replace)


def label(x):
    """ Returns the mnist number label corresponding to the array """
    if x.ndim == 1:
        return np.where(x>0)[0]
    else:
        return np.where(x>0)[1]


def idx_to_label(x, trainY):
    """ Returns a an array of labels given an array of indexes and training set """
    y_lab = label(trainY)
    return y_lab[x]
