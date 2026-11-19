import numpy as np
import sklearn
import torch

# def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
#     """Gaussian kernel with squared distance in exponential term replaced by EMD
#     Args:
#       x, y: 1D pmf of two distributions with the same support
#       sigma: standard deviation
#     """
#     support_size = max(len(x), len(y))
#     d_mat = toeplitz(range(support_size)).astype(np.float)
#     distance_mat = d_mat / distance_scaling

#     # convert histogram values x and y to float, and make them equal len
#     x = x.astype(np.float)
#     y = y.astype(np.float)
#     if len(x) < len(y):
#         x = np.hstack((x, [0.0] * (support_size - len(x))))
#     elif len(y) < len(x):
#         y = np.hstack((y, [0.0] * (support_size - len(y))))

#     emd = pyemd.emd(x, y, distance_mat)
#     return np.exp(-emd * emd / (2 * sigma * sigma))


# def gaussian_rbf(x, y, sigma=1.0):
#     lx, ly = len(x), len(y)
#     if lx != ly:
#         max_len = max(lx, ly)
#         if lx < max_len:
#             x = np.hstack((x, np.zeros(max_len - lx)))
#         else:  # ly < max_len
#             y = np.hstack((y, np.zeros(max_len - ly)))

#     squared_dist = np.sum((x - y) ** 2)

#     return np.exp(-squared_dist / (2 * sigma**2))


# def disc(samples1, samples2, kernel, *args, **kwargs):
#     """Discrepancy between 2 samples"""
#     d = 0
#     for s1 in samples1:
#         for s2 in samples2:
#             d += kernel(s1, s2, *args, **kwargs)
#     d /= len(samples1) * len(samples2)
#     return d


def mean_pairwise_distance(dists_GR):
    return np.sqrt(dists_GR.mean())


def get_pairwise_distances(generated_dataset, reference_dataset):
    return (
        sklearn.metrics.pairwise_distances(
            reference_dataset, generated_dataset, metric="euclidean"
        )
        ** 2
    )


def _off_diagonal_values(matrix):
    dists = np.asarray(matrix)
    mask = ~np.eye(dists.shape[0], dtype=bool)
    return dists[mask]


def get_sigmas(dists_GR):
    base_sigmas = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    mult_factor = mean_pairwise_distance(dists_GR)
    return base_sigmas * mult_factor


def compute_mmd(
    samples1,
    samples2,
    reduction="max",
    sigma_dist="GR",
    is_hist=True,
    sigmas=None,
    pad=True,
    verbose=False,
    return_components=False,
):
    """
    samples 1 is supposed to be generated samples, and samples 2 is supposed to be reference samples.
    """
    if len(samples1) <= 1 or len(samples2) <= 1:
        raise ValueError("samples1 and samples2 must have more than 1 sample each.")
    if isinstance(samples1, torch.Tensor):
        samples1 = samples1.detach().cpu().numpy()
    if isinstance(samples2, torch.Tensor):
        samples2 = samples2.detach().cpu().numpy()

    if reduction not in ["max", "mean", "sum"]:
        raise ValueError(
            f"reduction must be one of ['max', 'mean', 'sum'], but got {reduction}."
        )

    samples1 = np.asarray(samples1)
    samples2 = np.asarray(samples2)
    # if samples1.ndim != samples2.ndim:
    #     raise ValueError(
    #         f"samples1(dim: {samples1.ndim}) and samples2(dim: {samples2.ndim}) must have the same number of dimensions."
    #     )

    if not pad and samples1.shape[1] != samples2.shape[1]:
        raise ValueError(
            f"samples1(dim: {samples1.shape[1]}) and samples2(dim: {samples2.shape[1]}) must have the same number of features when pad=False."
        )

    if is_hist:
        sums1 = [np.sum(s1) for s1 in samples1]
        sums2 = [np.sum(s2) for s2 in samples2]

        if any(s == 0 for s in sums1) or any(s == 0 for s in sums2):
            raise ValueError(
                "All histograms must have a non-zero sum to be normalized into pmf."
            )
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]

    if pad:
        max_len = 0
        max_len = max(max_len, max(len(h) for h in samples1))
        max_len = max(max_len, max(len(h) for h in samples2))

        if max_len > 0:
            # Pad all histograms with trailing zeros to match the longest one.
            samples1 = [np.pad(h, (0, max_len - len(h)), "constant") for h in samples1]
            samples2 = [np.pad(h, (0, max_len - len(h)), "constant") for h in samples2]

    if np.array(samples1).ndim != 2 or np.array(samples2).ndim != 2:
        raise ValueError(
            f"After normalization and padding, samples1(dim: {np.asarray(samples1).ndim}) and samples2(dim: {np.asarray(samples2).ndim}) must be 2-dimensional arrays."
        )

    # https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py
    GG = get_pairwise_distances(samples1, samples1)
    GR = get_pairwise_distances(samples1, samples2)
    RR = get_pairwise_distances(samples2, samples2)

    mmds = []
    components = []
    result_mmd = float("inf")
    base_sigmas = np.array([0.25, 0.5, 1.0, 2.0, 4.0])

    if sigmas is None:
        if sigma_dist == "GR":
            sigma_base_dist = GR
        elif sigma_dist == "GG":
            sigma_base_dist = _off_diagonal_values(GG)
        elif sigma_dist == "RR":
            sigma_base_dist = _off_diagonal_values(RR)
        else:
            raise ValueError(
                f"sigma_dist must be one of ['GR', 'GG', 'RR'], but got {sigma_dist}."
            )

        sigmas = get_sigmas(sigma_base_dist)

    for sigma, base_sigma in zip(sigmas, base_sigmas):
        gamma = 1 / (2 * sigma**2)

        K_GR = np.exp(-gamma * GR)
        K_GG = np.exp(-gamma * GG)
        K_RR = np.exp(-gamma * RR)

        u_GG = (np.sum(K_GG) - np.trace(K_GG)) / (len(samples1) * (len(samples1) - 1))
        u_RR = (np.sum(K_RR) - np.trace(K_RR)) / (len(samples2) * (len(samples2) - 1))

        mmd = u_GG + u_RR - 2 * K_GR.mean()
        mmds.append(mmd)

        components.append(
            {
                "sigma": sigma,
                "base_sigma": base_sigma,
                "mmd": mmd,
                "u_GG": u_GG,
                "u_RR": u_RR,
                "K_GR": K_GR.mean(),
            }
        )
        if verbose:
            print(
                f"sigma: {sigma}, mmd: {mmd} (GG: {u_GG}, RR: {u_RR}, GR: {K_GR.mean()})"
            )

    if reduction == "max":
        result_mmd = max(mmds)
    elif reduction == "mean":
        result_mmd = np.mean(mmds)
    elif reduction == "sum":
        result_mmd = np.sum(mmds)

    mmd = max(result_mmd, 0.0)

    if return_components:
        return mmd, components

    return mmd


# def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
#     ''' MMD between two samples
#     '''
#     # normalize histograms into pmf
#     if is_hist:
#         samples1 = [s1 / np.sum(s1) for s1 in samples1]
#         samples2 = [s2 / np.sum(s2) for s2 in samples2]
#     # print('===============================')
#     # print('s1: ', disc(samples1, samples1, kernel, *args, **kwargs))
#     # print('--------------------------')
#     # print('s2: ', disc(samples2, samples2, kernel, *args, **kwargs))
#     # print('--------------------------')
#     # print('cross: ', disc(samples1, samples2, kernel, *args, **kwargs))
#     # print('===============================')
#     mmd = disc(samples1, samples1, kernel, *args, **kwargs) + disc(samples2, samples2, kernel, *args, **kwargs) - 2 * disc(samples1, samples2, kernel, *args, **kwargs)
#     return mmd
