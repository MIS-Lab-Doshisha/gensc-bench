import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_losses(train_losses: list, val_losses: list, output_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Linear plot
    ax1.plot(train_losses, label="Training Loss", color="blue")
    ax1.plot(val_losses, label="Validation Loss", color="orange")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid()
    ax1.text(
        0.5,
        -0.15,
        "Training and Validation Losses (Linear Scale)",
        ha="center",
        va="center",
        transform=ax1.transAxes,
    )

    # Log plot
    ax2.plot(train_losses, label="Training Loss", color="blue")
    ax2.plot(val_losses, label="Validation Loss", color="orange")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid()
    ax2.text(
        0.5,
        -0.15,
        "Training and Validation Losses (Log Scale)",
        ha="center",
        va="center",
        transform=ax2.transAxes,
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_plot.pdf")
    plt.close()


def is_binary_matrix(matrix):
    unique_values = np.unique(matrix)
    if len(unique_values) <= 2:
        return True
    return False


def plot_matrix_grid(
    matrices,
    output_path_name=None,
    grid_size=(None, None),
    column_titles=None,
    plot_title=None,
    cmap_heat="inferno",
    cmap_binary="gray",
    global_min=None,
    global_max=None,
    column_title_kwards={"fontsize": 16},
    title_kwards={"fontsize": 20},
    norm=None,
    cbar_ticks=None,
):
    n = len(matrices)
    cbar_kwards = {}

    if n == 0:
        raise ValueError("No matrices to plot")
    if global_min is None:
        global_min = min(matrix.min() for matrix in matrices)
    if global_max is None:
        global_max = max(matrix.max() for matrix in matrices)
    if global_min == global_max:
        global_max += 1

    if grid_size == (None, None):
        n_cols = math.ceil(math.sqrt(n))
        n_rows = math.ceil(n / n_cols)
    else:
        n_rows, n_cols = grid_size
        if n_rows * n_cols < n:
            raise ValueError(f"Grid size {grid_size} is too small for {n} matrices")

    if column_titles is not None:
        if len(column_titles) != n_cols:
            raise ValueError(
                f"Number of column titles {len(column_titles)} does not match number of columns {n_cols}"
            )

    fig = plt.figure(
        figsize=(n_cols * 3, n_rows * 3),
        # constrained_layout=True,
        dpi=300,
    )

    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1] * n_cols + [0.05],
        wspace=0.05,
        hspace=0.05,
    )
    axes = [
        fig.add_subplot(gs[i // n_cols, i % n_cols]) for i in range(n_rows * n_cols)
    ]

    cax = fig.add_subplot(gs[:, -1])

    if column_titles is not None:
        for j, title in enumerate(column_titles):
            axes[j].set_title(title, **column_title_kwards)

    if plot_title is not None:
        fig.suptitle(plot_title, **title_kwards, y=0.96)

    for i, matrix in enumerate(matrices):
        ax = axes[i]
        if not is_binary_matrix(matrix):
            cmap = plt.get_cmap(cmap_heat)
            cmap.set_over("cyan")
            cbar_kwards["extend"] = "max"

            heatmap_plot = sns.heatmap(
                matrix,
                annot=False,
                cmap=cmap,
                cbar=False,
                ax=ax,
                square=True,
                linewidths=0.0,
                rasterized=True,
                vmin=global_min if norm is None else None,
                vmax=global_max if norm is None else None,
                norm=norm,
            )
        else:
            heatmap_plot = sns.heatmap(
                matrix,
                annot=False,
                cmap=cmap_binary,
                cbar=False,
                ax=ax,
                square=True,
                linewidths=0.0,
                rasterized=True,
            )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    mappable = heatmap_plot.collections[0]
    cbar = fig.colorbar(mappable, cax=cax, **cbar_kwards)

    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)

    # plt.tight_layout()
    output_dir = Path(output_path_name).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_path_name).with_suffix(".pdf"), bbox_inches="tight")
    plt.show()
    plt.close()


def get_formatted_dataset_name(key: str, abbr=False) -> str:
    if abbr:
        name_mapping = {
            "ba": "BA",
            "planar": "Planar",
            "sbm": "SBM",
            "wba": "WSF",
            "ws": "WS",
            "sc": "SC",
            "hcp": "HCP",
            "nimh": "NIMH",
        }
    else:
        name_mapping = {
            "ba": "Barabási-Albert",
            "planar": "Planar",
            "sbm": "Stochastic Block Model",
            "wba": "Weighted Scale-Free",
            "ws": "Watts-Strogatz",
            "sc": "Structural Connectivity",
            "hcp": "Human Connectome Project",
        }
    return name_mapping.get(key, key)


def get_formatted_model_name(key: str) -> str:
    name_mapping = {
        "vae": "VAE",
        "gan": "WGAN-GP",
        "ddpm": "DDPM",
        "real": "Real",
    }

    return name_mapping.get(key, key)


def get_formatted_metric_name(key: str) -> str:
    name_mapping = {
        "Planarity Rate": "Planarity",
        "Degree MMD": "Deg. MMD",
        "Clustering MMD": "Clus. MMD",
        "Orbit MMD": "Orbit MMD",
        "Spectral MMD": "Spec. MMD",
        "Embedding MMD": "Emb. MMD",
        "Strength MMD": "Str. MMD",
    }

    return name_mapping.get(key, key)


def format_sci(value):
    s = f"{value:.1e}"
    return re.sub(r"e([+-])0(\d)$", r"e\1\2", s)


def format_mean_sd(mean, sd, count):
    if pd.isna(mean):
        return "-"

    if count <= 1 or pd.isna(sd):
        return format_sci(mean)

    return f"{format_sci(mean)} ± {format_sci(sd)}"
