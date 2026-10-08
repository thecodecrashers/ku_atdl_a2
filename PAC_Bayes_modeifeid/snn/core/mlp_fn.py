import numpy as np
import torch
import torch.nn.functional as F


def _trunc_normal(shape, std, generator=None):
    """ Truncated normal (resampled outside of 2 std), like tf.truncated_normal_initializer. """
    t = torch.empty(shape)
    # trunc_normal_ has no generator argument, so rely on the global torch seed.
    torch.nn.init.trunc_normal_(t, mean=0.0, std=std, a=-2 * std, b=2 * std)
    return t.numpy()


def multilayer_perceptron_params(layers):
    """ Random initial parameters [W1, b1, W2, b2, ...] (numpy, W has shape [n_in, n_out]). """
    # Preserve the upstream initialization distribution, hidden/output bias
    # convention, and TensorFlow matrix layout. This allows existing snapshots
    # to retain the same interpretation after the PyTorch migration; the two
    # frameworks' RNG implementations need not produce bit-identical draws.
    params = []
    for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:])):
        params.append(_trunc_normal([n_in, n_out], 0.04))
        params.append(np.full([n_out], 0.1 if i == 0 else 0.0, dtype=np.float32))
    return params


def multilayer_perceptron(x, params):
    """ Fully connected feedforward network with RELU activation.
    :param x: [batch, n_in] tensor
    :param params: list of tensors [W1, b1, W2, b2, ...]
    """
    # Keep W in [n_in,n_out] order instead of adopting torch.nn.Linear's
    # transposed storage. Hidden layers use ReLU and the final output is linear,
    # matching the original MLP operations and saved parameter ordering.
    next_layer = torch.matmul(x, params[0]) + params[1]
    for w, b in zip(params[2::2], params[3::2]):
        next_layer = torch.matmul(F.relu(next_layer), w) + b
    return next_layer


def MLP_withnoise(x, noise_list, params_mean_values):
    """ The same network where every parameter is perturbed by the corresponding entry of noise_list. """
    # Unlike upstream's function, this does not create fresh graph variables
    # from numeric initializers. It directly receives the live mean tensors;
    # noise_list already contains sigma*epsilon from Network._perturbations.
    # Both weights and biases are sampled, with gradients flowing through the
    # addition to any trainable mean and standard-deviation parameters.
    noisy = [m + n for m, n in zip(params_mean_values, noise_list)]
    return multilayer_perceptron(x, noisy)


def weight_diff(w1, w2):
    """ Calculates the array of differences between the weights in arrays """
    # Expand and flatten arrays
    _w1 = np.hstack([x.flatten() for x in w1])
    _w2 = np.hstack([x.flatten() for x in w2])
    return _w1 - _w2


def l2_norm(w1, w2):
    return np.linalg.norm(weight_diff(w1, w2))
