from typing import Literal, overload

import numpy as np
import torch
import torch.nn.functional as F


def to_tensor(data, device=None):
    if isinstance(data, torch.Tensor):
        return data.to(device) if device is not None else data
    return torch.as_tensor(data, device=device)


def mean_pairwise_distance(dists):
    return torch.sqrt(dists.mean())


def get_pairwise_distances(x, y):
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1).unsqueeze(0)

    return (x2 + y2 - 2.0 * (x @ y.T)).clamp_min_(0.0)


def _off_diagonal_values(matrix):
    n = matrix.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def get_sigmas(dists):
    base_sigmas = torch.tensor(
        [0.25, 0.5, 1.0, 2.0, 4.0], device=dists.device, dtype=dists.dtype
    )
    mult_factor = mean_pairwise_distance(dists)
    return base_sigmas * mult_factor


@overload
def compute_mmd(
    samples1: torch.Tensor | np.ndarray,
    samples2: torch.Tensor | np.ndarray,
    reduction: str = "max",
    sigma_dist: str = "GR",
    is_hist: bool = True,
    sigmas: torch.Tensor | None = None,
    pad: bool = True,
    verbose: bool = False,
    *,
    return_components: Literal[False] = False,
    device: str | torch.device | None = None,
) -> float: ...


@overload
def compute_mmd(
    samples1: torch.Tensor | np.ndarray,
    samples2: torch.Tensor | np.ndarray,
    reduction: str = "max",
    sigma_dist: str = "GR",
    is_hist: bool = True,
    sigmas: torch.Tensor | None = None,
    pad: bool = True,
    verbose: bool = False,
    *,
    return_components: Literal[True],
    device: str | torch.device | None = None,
) -> tuple[float, list[dict]]: ...


def compute_mmd(
    samples1,
    samples2,
    reduction="max",
    sigma_dist="GR",
    is_hist=True,
    sigmas=None,
    pad=True,
    verbose=False,
    *,
    return_components=False,
    device=None,
):
    """
    samples 1 is supposed to be generated samples, and samples 2 is supposed to be reference samples.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if len(samples1) <= 1 or len(samples2) <= 1:
        raise ValueError("samples1 and samples2 must have more than 1 sample each.")

    if reduction not in ["max", "mean", "sum"]:
        raise ValueError(
            f"reduction must be one of ['max', 'mean', 'sum'], but got {reduction}."
        )

    if pad:
        samples1 = [to_tensor(s, device=device).squeeze() for s in samples1]

        samples2 = [to_tensor(s, device=device).squeeze() for s in samples2]

        sample1_max_len = max(len(s) for s in samples1)
        sample2_max_len = max(len(s) for s in samples2)

        max_len = max(sample1_max_len, sample2_max_len)

        samples1 = torch.stack([F.pad(s, (0, max_len - len(s))) for s in samples1])
        samples2 = torch.stack([F.pad(s, (0, max_len - len(s))) for s in samples2])

    else:
        samples1 = to_tensor(samples1, device=device)
        samples2 = to_tensor(samples2, device=device)

        if samples1.shape[1] != samples2.shape[1]:
            raise ValueError(
                f"samples1(dim: {samples1.shape[1]}) and samples2(dim: {samples2.shape[1]}) must have the same number of features when pad=False."
            )

    if samples1.ndim != samples2.ndim:
        raise ValueError(
            f"samples1(dim: {samples1.ndim}) and samples2(dim: {samples2.ndim}) must have the same number of dimensions."
        )

    if is_hist:
        sums1 = samples1.sum(dim=1, keepdim=True)
        sums2 = samples2.sum(dim=1, keepdim=True)

        if torch.any(sums1 == 0) or torch.any(sums2 == 0):
            raise ValueError(
                "All histograms must have a non-zero sum to be normalized into pmf."
            )

        samples1 = samples1 / sums1
        samples2 = samples2 / sums2

    GG = get_pairwise_distances(samples1, samples1)
    GR = get_pairwise_distances(samples1, samples2)
    RR = get_pairwise_distances(samples2, samples2)

    if sigmas is None:
        if sigma_dist == "GR":
            sigmas = get_sigmas(GR)
        elif sigma_dist == "GG":
            sigmas = get_sigmas(GG)
        elif sigma_dist == "RR":
            sigmas = get_sigmas(RR)
        else:
            raise ValueError(
                f"sigma_dist must be one of ['GR', 'GG', 'RR'], but got {sigma_dist}."
            )

    else:
        sigmas = to_tensor(sigmas, device=device)

    gammas = 1.0 / (2.0 * sigmas.square())

    K_GG = torch.exp(-gammas[:, None, None] * GG[None, :, :])
    K_GR = torch.exp(-gammas[:, None, None] * GR[None, :, :])
    K_RR = torch.exp(-gammas[:, None, None] * RR[None, :, :])

    n1 = samples1.shape[0]
    n2 = samples2.shape[0]

    diag_GG = torch.diagonal(K_GG, dim1=1, dim2=2).sum(dim=1)
    diag_RR = torch.diagonal(K_RR, dim1=1, dim2=2).sum(dim=1)

    u_GG = (K_GG.sum(dim=(1, 2)) - diag_GG) / (n1 * (n1 - 1))
    u_RR = (K_RR.sum(dim=(1, 2)) - diag_RR) / (n2 * (n2 - 1))

    mmds = u_GG + u_RR - 2 * K_GR.mean(dim=(1, 2))

    if verbose:
        for i in range(len(sigmas)):
            print(
                f"sigma: {sigmas[i].item():.4f}, "
                f"MMD: {mmds[i].item():.6f}, "
                f"u_GG: {u_GG[i].item():.6f},"
                f"u_RR: {u_RR[i].item():.6f}, "
                f"K_GR: {K_GR[i].mean().item():.6f}"
            )

    if reduction == "max":
        result_mmd = mmds.max()
    elif reduction == "mean":
        result_mmd = mmds.mean()
    elif reduction == "sum":
        result_mmd = mmds.sum()

    result_mmd = torch.clamp(result_mmd, min=0.0)

    if return_components:
        base_sigmas = torch.tensor(
            [0.25, 0.5, 1.0, 2.0, 4.0], device=device, dtype=sigmas.dtype
        )

        components = [
            {
                "sigma": sigmas[i].item(),
                "base_sigma": base_sigmas[i].item(),
                "mmd": mmds[i].item(),
                "u_GG": u_GG[i].item(),
                "u_RR": u_RR[i].item(),
                "K_GR": K_GR[i].mean().item(),
            }
            for i in range(len(sigmas))
        ]
        return result_mmd.item(), components

    return result_mmd.item()
