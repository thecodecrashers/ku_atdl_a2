import numpy as np
import torch
import torch.nn.functional as F

from snn.core.mlp_fn import _trunc_normal, weight_diff, l2_norm  # noqa: F401  (weight_diff / l2_norm re-exported)

NUM_CLASSES = 10


def convolutional_net_params(scopes_list=None):
    """ Random initial parameters of the CIFAR-10 model (numpy, TF layout: conv kernels are [H, W, in, out]). """
    return [_trunc_normal([5, 5, 3, 64], 5e-2), np.zeros([64], dtype=np.float32),
            _trunc_normal([5, 5, 64, 64], 5e-2), np.full([64], 0.1, dtype=np.float32),
            _trunc_normal([4096, 384], 0.04), np.full([384], 0.1, dtype=np.float32),
            _trunc_normal([384, 192], 0.04), np.full([192], 0.1, dtype=np.float32),
            _trunc_normal([192, NUM_CLASSES], 1 / 192.0), np.zeros([NUM_CLASSES], dtype=np.float32)]


def _conv(x, kernel, bias):
    # kernel: [H, W, in, out] -> torch [out, in, H, W]; 5x5 stride 1 'SAME' padding
    return F.conv2d(x, kernel.permute(3, 2, 0, 1), bias, stride=1, padding=2)


def _pool(x):
    # 3x3 max pooling with stride 2 and TensorFlow 'SAME' padding (one extra row/column of padding at the end)
    return F.max_pool2d(F.pad(x, (0, 1, 0, 1), value=float("-inf")), kernel_size=3, stride=2)


def _lrn(x):
    # tf.nn.lrn(depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75); torch divides alpha by the window size (9)
    return F.local_response_norm(x, size=9, alpha=0.001, beta=0.75, k=1.0)


def convolutional_net(images, params):
    """ The CIFAR-10 model.
    :param images: [batch, 32, 32, 3] (channels last) tensor
    :param params: list of 10 tensors [conv1 W, b, conv2 W, b, local3 W, b, local4 W, b, softmax W, b]
    :return: logits
    """
    x = images.permute(0, 3, 1, 2)
    conv1 = F.relu(_conv(x, params[0], params[1]))
    norm1 = _lrn(_pool(conv1))
    conv2 = F.relu(_conv(norm1, params[2], params[3]))
    pool2 = _pool(_lrn(conv2))
    # Back to channels last so that the flattened layout matches the original weights
    reshape = pool2.permute(0, 2, 3, 1).reshape(-1, 4096)
    local3 = F.relu(torch.matmul(reshape, params[4]) + params[5])
    local4 = F.relu(torch.matmul(local3, params[6]) + params[7])
    return torch.matmul(local4, params[8]) + params[9]


def CNN_withnoise(images, noise_list, params_mean_values):
    noisy = [m + n for m, n in zip(params_mean_values, noise_list)]
    return convolutional_net(images, noisy)
