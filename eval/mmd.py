import numpy as np
import sklearn
import torch

def gaussian_rbf(x, y, sigma=1.0):
    lx, ly = len(x), len(y)
    if lx != ly:
        max_len = max(lx, ly)
        if lx < max_len:
            x = np.hstack((x, np.zeros(max_len - lx)))
        else:  # ly < max_len
            y = np.hstack((y, np.zeros(max_len - ly)))

    squared_dist = np.sum((x - y) ** 2)

    return np.exp(-squared_dist / (2 * sigma**2))


def disc(samples1, samples2, kernel, *args, **kwargs):
    """Discrepancy between 2 samples"""
    d = 0
    for s1 in samples1:
        for s2 in samples2:
            d += kernel(s1, s2, *args, **kwargs)
    d /= len(samples1) * len(samples2)
    return d


def mean_pairwise_distance(dists_GR):
    return np.sqrt(dists_GR.mean())


def get_pairwise_distances(generated_dataset, reference_dataset):
    return (
        sklearn.metrics.pairwise_distances(
            reference_dataset, generated_dataset, metric="euclidean", n_jobs=8
        )
        ** 2
    )


def get_sigmas(dists_GR):
    base_sigmas = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    mult_factor = mean_pairwise_distance(dists_GR)
    return base_sigmas * mult_factor


def compute_mmd(samples1, samples2, is_hist=True, verbose=False):
    if len(samples1) <= 1 or len(samples2) <= 1:
        raise ValueError("samples1 and samples2 must have more than 1 sample each.")
    if isinstance(samples1, torch.Tensor):
        samples1 = samples1.cpu().numpy()
    if isinstance(samples2, torch.Tensor):
        samples2 = samples2.cpu().numpy()
    # if samples1.ndim != samples2.ndim:
    #     raise ValueError(
    #         f"samples1(dim: {samples1.ndim}) and samples2(dim: {samples2.ndim}) must have the same number of dimensions."
    #     )

    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]

    max_len = 0
    max_len = max(max_len, max(len(h) for h in samples1))
    max_len = max(max_len, max(len(h) for h in samples2))

    if max_len > 0:
        samples1 = [np.pad(h, (0, max_len - len(h)), "constant") for h in samples1]
        samples2 = [np.pad(h, (0, max_len - len(h)), "constant") for h in samples2]

    if np.array(samples1).ndim != 2 or np.array(samples2).ndim != 2:
        raise ValueError(
            f"After normalization and padding, samples1(dim: {np.ndarray(samples1).ndim}) and samples2(dim: {np.ndarray(samples2).ndim}) must be 2-dimensional arrays."
        )

    # https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py
    GG = get_pairwise_distances(samples1, samples1)
    GR = get_pairwise_distances(samples1, samples2)
    RR = get_pairwise_distances(samples2, samples2)

    max_mmd = 0
    sigmas = get_sigmas(GR)
    for sigma in sigmas:
        gamma = 1 / (2 * sigma**2)

        K_GR = np.exp(-gamma * GR)
        K_GG = np.exp(-gamma * GG)
        K_RR = np.exp(-gamma * RR)

        u_GG = (np.sum(K_GG) - np.trace(K_GG)) / (len(samples1) * (len(samples1) - 1))
        u_RR = (np.sum(K_RR) - np.trace(K_RR)) / (len(samples2) * (len(samples2) - 1))

        mmd = u_GG + u_RR - 2 * K_GR.mean()
        max_mmd = mmd if mmd > max_mmd else max_mmd
        if verbose:
            print(f"sigma: {sigma}, mmd: {mmd}")

    return max(max_mmd, 0.0)

