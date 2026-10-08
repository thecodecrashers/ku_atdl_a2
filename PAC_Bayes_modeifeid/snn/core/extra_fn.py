import numpy as np
import random
import math


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


def gaussian_kl(mean, prior_mean, log_post_std, log_prior_std):
    """Evaluate diagonal-to-isotropic Gaussian KL in float64 without cancellation."""
    # Analytic KL for Q=N(mu,diag(sigma^2)) and P=N(w0,lambda*I), with
    # log_post_std=log(sigma), log_prior_std=log(sqrt(lambda)).
    # Use the supplied final means, not the initialization snapshot. Float64
    # and expm1(r)-r reduce cancellation when sigma^2 is close to lambda.
    # No posterior sampling is needed to compute this quantity.
    if not len(mean) == len(prior_mean) == len(log_post_std):
        raise ValueError("Posterior and prior parameter lists must have matching lengths")
    rho = float(log_prior_std)
    value = 0.0
    for w, w0, log_std in zip(mean, prior_mean, log_post_std):
        w, w0, log_std = (np.asarray(a, dtype=np.float64) for a in (w, w0, log_std))
        if w.shape != w0.shape or w.shape != log_std.shape:
            raise ValueError("Posterior and prior parameter shapes must match")
        r = 2 * (log_std - rho)
        value += np.sum(np.expm1(r) - r) + np.sum((w - w0) ** 2) * math.exp(-2 * rho)
    value = float(value / 2)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Gaussian KL must be finite and nonnegative")
    return value


def discretized_prior_bound(mean, prior_mean, log_post_std, log_prior_std, m,
                            delta=0.025, precision=100.0, base=0.1):
    """Select the better neighboring positive-index prior from the paper's grid."""
    # The prior grid is lambda_j=base*exp(-j/precision), j>=1. Convert the
    # continuous training prior to its two neighboring positive integer indices
    # and compute the actual Gaussian KL for each candidate prior.
    # Replacing a negative j by abs(j) would change which variance the index
    # represents; reject out-of-domain legacy models instead of certifying them.
    if m <= 1 or not 0 < delta < 1 or precision <= 0 or base <= 0:
        raise ValueError("Invalid PAC-Bayes bound parameters")
    j = precision * (math.log(base) - 2 * float(log_prior_std))
    if not math.isfinite(j) or j < 1 - 1e-4:
        raise ValueError("Prior variance is outside the positive-index grid; retraining is required")
    j = max(1.0, j)
    candidates = []
    for index in sorted({max(1, math.floor(j)), max(1, math.ceil(j))}):
        rho = (math.log(base) - index / precision) / 2
        kl = gaussian_kl(mean, prior_mean, log_post_std, rho)
        # The 2*log(j) term pays for selecting j using the union-bound weights
        # 6/(pi^2*j^2). This entropy budget C enters kl_inverse(q_upper,C);
        # B=sqrt(C/2) is the training complexity term, not an error probability.
        relative_entropy = (kl + math.log(math.pi ** 2 * m / (6 * delta))
                            + 2 * math.log(index)) / (m - 1)
        candidates.append({"prior grid index": index, "selected log prior std": rho,
                           "selected prior variance": math.exp(2 * rho),
                           "KL(Q || P)": kl, "relative entropy bound": relative_entropy,
                           "generalization/complexity term B": math.sqrt(relative_entropy / 2)})
    return min(candidates, key=lambda item: item["relative entropy bound"])


def generate_noise(layer_shapes):
    noise_list = []
    for l_shape in layer_shapes:
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
    for (n_in,n_out) in zip(layers[:-1],layers[1:]):
        N += n_in*n_out
        N += n_out
    return N


def KLdiv(pbar,p):
    return pbar * np.log(pbar/p) + (1-pbar) * np.log((1-pbar)/(1-p))


def KLdiv_prime(pbar,p):
    return (1-pbar)/(1-p) - pbar/p


def Newt(p,q,c):
    newp = p - (KLdiv(q,p) - c)/KLdiv_prime(q,p)
    return newp


def approximate_BPAC_bound(train_accur, B_init, niter=5):
    """Invert the PAC-Bayes KL bound; niter is retained for API compatibility."""
    # Legacy callers provide accuracy and B, so recover q=1-accuracy and C=2B^2.
    # niter no longer chooses a Newton iteration count. This wrapper alone does
    # not include MC uncertainty; the final evaluator explicitly applies the
    # first inversion to q before performing the second PAC inversion.
    if not math.isfinite(float(B_init)) or B_init < 0:
        raise ValueError("B_init must be finite and nonnegative")
    return inverse_binary_kl(1 - train_accur, 2 * B_init ** 2)


def hoeffdingbnd(M,delta):
    eps = np.sqrt(np.log(2/delta)/M)
    return eps


def SamplesConvBound(train_error=0.028,M=1000,delta=0.01,p_init = None, niter = 5):
    # Preserve the upstream helper's return convention: the additional error
    # margin, not the corrected error itself. p_init/niter are compatibility
    # arguments; the new evaluator calls monte_carlo_error_upper directly.
    p_next = monte_carlo_error_upper(train_error, M, delta)
    print("Chernoff's error", p_next-train_error)
    return p_next-train_error


def next_batch(xx,yy, batchsize, idx):
    return [xx[idx*batchsize:(idx+1)*batchsize], yy[idx*batchsize:(idx+1)*batchsize]]


def shuffledata(xx,yy):
    ns = xx.shape[0]
    idx = random.sample(list(np.arange(ns)), ns)
    xx = xx[idx,:]
    yy = yy[idx,:]
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


def create_label_dictionary(yy):
    """ Creates the label dictionary used to sample data points within the priors dataset"""
    labels = np.where(yy > 0)[1]
    idxs = range(0, yy.shape[0])
    dict = {}
    for x, y in zip(idxs, labels):
        dict.setdefault(y,[]).append(x)
    return dict
