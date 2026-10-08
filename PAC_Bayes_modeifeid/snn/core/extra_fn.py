import math
import random

import numpy as np


def inverse_binary_kl(q, c):
    """Return a conservative upper solution of binary KL(q || p) <= c."""
    # New common solver for both the MC and PAC inversions. Binary kl(q||p)
    # here is between Bernoulli error probabilities; it is distinct from the
    # high-dimensional Gaussian KL(Q||P) computed by gaussian_kl below.
    # Upstream uses five unbracketed Newton steps and returns 1 immediately
    # if its initial q+B exceeds 1. Bisection instead solves on p in [q, 1]
    # without that initialization-dependent early return.
    q, c = float(q), float(c)
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("q must be finite and in [0, 1]")
    if math.isnan(c) or c < 0:
        raise ValueError("c must be nonnegative")
    if q == 1 or math.isinf(c):
        return 1.0
    if c == 0:
        return q
    if q == 0:
        # kl(0||p) = -log(1-p), so the endpoint has an exact solution.
        # expm1 is accurate for small c; nextafter rounds conservatively upward.
        return min(1.0, math.nextafter(-math.expm1(-c), 1.0))
    lo, hi = q, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        kl = q * math.log(q / mid) + (1 - q) * (math.log1p(-q) - math.log1p(-mid))
        if kl > c:
            hi = mid
        else:
            lo = mid
    # Keep the upper bracket rather than the midpoint/lower bracket, so finite
    # numerical precision does not underestimate the mathematical upper endpoint.
    return hi


def monte_carlo_error_upper(error, samples, delta=0.01):
    """Bound the expected empirical error using independent posterior draws."""
    # Implements the first inversion in the paper's Section 3.3 / Eq. (6).
    # error is the mean full-training-set error of N independently drawn SNNs.
    # The KL Chernoff argument also applies to these bounded [0,1] draw errors.
    # Counting all examples as independent MC draws would make the penalty
    # artificially too small, because examples within a draw share one network.
    # delta controls failure probability; N controls tightness at that delta.
    if samples < 1 or int(samples) != samples:
        raise ValueError("samples must be a positive integer")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    return inverse_binary_kl(error, math.log(2 / delta) / samples)


def generate_noise(layer_shapes, laplace=False):
    noise_list = []
    for l_shape in layer_shapes:
        if laplace:
            noise_list.append(np.random.laplace(size=l_shape).astype(np.float32))
        else:
            noise_list.append(np.random.normal(size=l_shape).astype(np.float32))
    return noise_list


def generate_zero_noise(layer_shapes):
    noise_list = []
    for l_shape in layer_shapes:
        noise_list.append(np.zeros(l_shape, dtype=np.float32))
    return noise_list


def margin_loss(yhat, y):
    mls = []
    for row in range(len(y)):
        j = np.argmax(y[row])
        mls.append(yhat[row][j] - max([i for i in range(len(yhat[row])) if i != j]))
    return mls


def count_MLP_params(layers):
    """
    For computing VC dimension bound with boundVCdim()
    """
    N = 0
    for n_in, n_out in zip(layers[:-1], layers[1:]):
        N += n_in * n_out
        N += n_out
    return N


def KLdiv(pbar, p):
    return pbar * np.log(pbar / p) + (1 - pbar) * np.log((1 - pbar) / (1 - p))


def KLdiv_prime(pbar, p):
    return (1 - pbar) / (1 - p) - pbar / p


def Newt(p, q, c):
    newp = p - (KLdiv(q, p) - c) / KLdiv_prime(q, p)
    return newp


def approximate_BPAC_bound(train_accur, B_init, niter=5):
    B_RE = 2 * B_init**2
    A = 1 - train_accur
    B_next = B_init + A
    if B_next > 1.0:
        return 1.0
    for i in range(niter):
        B_next = Newt(B_next, A, B_RE)
    return B_next


def hoeffdingbnd(M, delta):
    eps = np.sqrt(np.log(2 / delta) / M)
    return eps


def SamplesConvBound(train_error=0.028, M=1000, delta=0.01, p_init=None, niter=5):
    c = np.log(2 / delta) / M
    if p_init is None:
        p_init = hoeffdingbnd(M, delta)
        print("Hoeffding's error", p_init)
    p_next = p_init + train_error
    for i in range(niter):
        p_next = Newt(p_next, train_error, c)
    print("Chernoff's error", p_next - train_error)
    return p_next - train_error


def next_batch(xx, yy, batchsize, idx):
    return [
        xx[idx * batchsize : (idx + 1) * batchsize],
        yy[idx * batchsize : (idx + 1) * batchsize],
    ]


def shuffledata(xx, yy):
    ns = xx.shape[0]
    idx = random.sample(list(np.arange(ns)), ns)
    xx = xx[idx, :]
    yy = yy[idx, :]
    return xx, yy


def sort_corresponding_to_labels(trainX, trainY, label_order):
    """Sorts the x and y such that they match the training order in label order. Tested."""
    trainY_labels = np.where(trainY > 0)[1]

    idxs = np.arange(0, len(trainY_labels))
    idmap = dict((id, pos) for pos, id in enumerate(label_order))
    l_ordered = [
        (x, y, _, _o)
        for _, _o, x, y in sorted(
            zip(idxs, trainY_labels, trainX, trainY), key=lambda x: idmap[x[1]]
        )
    ]
    newlabord = [l_ordered[i][3] for i in range(len(l_ordered))]
    x_ordered = np.array([l_ordered[i][0] for i in range(len(l_ordered))])
    y_ordered = np.array([l_ordered[i][1] for i in range(len(l_ordered))])

    return x_ordered, y_ordered, newlabord


def concat_labels(label_order):
    return np.hstack(label_order)


def label_indices(x, L):
    """Returns the indices of data points with label L. TESTED."""
    labels = np.where(x > 0)[1]
    return np.where(labels == L)[0]


def sample_label(x, L):
    """Samples a random instance with label L. Returns the index of the data point location."""
    idx = label_indices(x, L)
    return np.random.choice(idx)


def sample_label_from_dict(dd, L, size=None, replace=True):
    if isinstance(L, np.ndarray):
        return np.random.choice(dd[L[0]], size=size, replace=replace)
    else:
        return np.random.choice(dd[L], size=size, replace=replace)


def label(x):
    """Returns the mnist number label corresponding to the array"""
    if x.ndim == 1:
        return np.where(x > 0)[0]
    else:
        return np.where(x > 0)[1]


def idx_to_label(x, trainY):
    """Returns a an array of labels given an array of indexes and training set"""
    y_lab = label(trainY)
    return y_lab[x]


def create_label_dictionary(yy):
    """Creates the label dictionary used to sample data points within the priors dataset"""
    labels = np.where(yy > 0)[1]
    idxs = range(0, yy.shape[0])
    dict = {}
    for x, y in zip(idxs, labels):
        dict.setdefault(y, []).append(x)
    return dict
