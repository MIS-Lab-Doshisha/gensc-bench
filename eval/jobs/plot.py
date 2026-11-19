import collections
import math
import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cmcrameri import cm
from matplotlib.colors import TwoSlopeNorm, to_rgba
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory
from scipy.stats import pearsonr

from common.graph_util import get_top_n_edges, matrix_to_vec
from common.visualize import (
    get_formatted_dataset_name,
    get_formatted_model_name,
    is_binary_matrix,
)


def _get_common_scale(target_mean, comp_means_dict):
    all_matrices = [target_mean] + list(comp_means_dict.values())
    all_flat = np.concatenate([mat.flatten() for mat in all_matrices])

    global_min = np.min(all_flat)
    global_max = np.max(all_flat)
    p99_max = np.percentile(all_flat, 99.96)

    if p99_max * 2.0 < global_max:
        global_max = p99_max
    if global_min == global_max:
        global_min -= 0.5
        global_max += 0.5
    return global_min, global_max


def _get_common_scale_symmetric(diff_dict):
    all_diffs = np.concatenate([mat.flatten() for mat in diff_dict.values()])
    max_abs = np.percentile(np.abs(all_diffs), 99)

    if max_abs == 0:
        max_abs = 1.0

    return -max_abs, max_abs


def plot_mean_matrices(
    target_matrices,
    comparison_groups,
    output_path,
    cmap_mean="gist_heat",
    cmap_diff=cm.vik,
    norm=None,
    cbar_ticks_mean=None,
    cbar_ticks_diff=None,
    diff_vmin=None,
    diff_vmax=None,
):
    """
    Plots the mean matrices of the target and comparison groups, along with their differences.
    target_matrices: list of np.ndarray
    comparison_groups: dict of {label: list of np.ndarray}
    output_path: str or Path

    """

    target_mean = np.mean(target_matrices, axis=0)

    comp_means = {
        label: np.mean(matrices, axis=0)
        for label, matrices in comparison_groups.items()
    }
    diff_means = {
        label: target_mean - comp_mean for label, comp_mean in comp_means.items()
    }

    vmin, vmax = _get_common_scale(target_mean, comp_means)
    if diff_vmin is None or diff_vmax is None:
        diff_vmin, diff_vmax = _get_common_scale_symmetric(diff_means)

    num_cols = len(comp_means) + 1

    fig, axes = plt.subplots(
        2, num_cols, figsize=(4 * num_cols, 8), squeeze=False, constrained_layout=True
    )
    cmap_mean = plt.get_cmap(cmap_mean)
    cmap_mean.set_over("white")
    cmap_diff = plt.get_cmap(cmap_diff)
    cmap_diff.set_over(cmap_diff(1.0))
    cmap_diff.set_under(cmap_diff(0.0))
    # cmap_diff.set_under("#002E2A")
    # cmap_diff.set_over("#180023")

    ax = axes[0, 0]
    im_mean = ax.imshow(
        target_mean,
        cmap=cmap_mean,
        vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None,
        norm=norm,
    )
    # ax.set_title("Target Mean Matrix", fontsize=16, y=-0.1)
    ax.set_aspect("equal")
    ax.axis("off")

    for i, (label, comp_mean) in enumerate(comp_means.items()):
        ax = axes[0, i + 1]
        ax.imshow(
            comp_mean,
            cmap=cmap_mean,
            vmin=vmin if norm is None else None,
            vmax=vmax if norm is None else None,
            norm=norm,
        )
        # ax.set_title(f"{label} Mean Matrix", fontsize=16, y=-0.1)
        ax.set_aspect("equal")
        ax.axis("off")

    axes[1, 0].axis("off")

    diff_norm = TwoSlopeNorm(vmin=diff_vmin, vcenter=0, vmax=diff_vmax)

    for i, (label, diff_mean) in enumerate(diff_means.items()):
        ax = axes[1, i + 1]
        im_diff = ax.imshow(diff_mean, cmap=cmap_diff, norm=diff_norm)
        # ax.set_title(f"Target - {label}", fontsize=16, y=-0.1)
        ax.set_aspect("equal")
        ax.axis("off")

    print("=== Mean matrices ===")
    print(
        "target:",
        "min =",
        np.nanmin(target_mean),
        "max =",
        np.nanmax(target_mean),
        "finite =",
        np.isfinite(target_mean).all(),
    )

    for label, comp_mean in comp_means.items():
        print(
            label,
            "min =",
            np.nanmin(comp_mean),
            "max =",
            np.nanmax(comp_mean),
            "finite =",
            np.isfinite(comp_mean).all(),
        )

    print("common mean scale:", vmin, vmax)
    print("norm:", norm)

    print("\n=== Difference matrices ===")
    for label, diff_mean in diff_means.items():
        print(
            label,
            "min =",
            np.nanmin(diff_mean),
            "max =",
            np.nanmax(diff_mean),
            "finite =",
            np.isfinite(diff_mean).all(),
        )

    print("diff scale:", diff_vmin, diff_vmax)

    cbar_mean = fig.colorbar(
        im_mean, ax=axes[0, :], fraction=0.05, pad=0.01, extend="max"
    )
    cbar_mean.set_label("Mean Value", rotation=270, labelpad=15, fontsize=18)
    cbar_mean.ax.tick_params(labelsize=16)
    if cbar_ticks_mean is not None:
        cbar_mean.set_ticks(cbar_ticks_mean)

    cbar_diff = fig.colorbar(
        im_diff, ax=axes[1, :], fraction=0.05, pad=0.01, extend="both"
    )
    cbar_diff.set_label("Difference", rotation=270, labelpad=15, fontsize=18)
    cbar_diff.ax.tick_params(labelsize=16)

    for patch in cbar_diff.ax.patches:
        patch.set_linewidth(0)
    if cbar_ticks_diff is not None:
        cbar_diff.set_ticks(cbar_ticks_diff)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved mean matrix comparison to {output_path}")


def get_degree_distribution_points(degrees, log_binning=False, num_bins=20):
    if not degrees:
        return [], []

    degree_counts = collections.Counter(d for d in degrees if d > 0)

    if not degree_counts:
        return [], []

    unique_degrees, counts = zip(*sorted(degree_counts.items()))

    if not log_binning:
        return list(unique_degrees), list(counts)

    min_deg, max_deg = min(unique_degrees), max(unique_degrees)
    if min_deg == max_deg:
        return list(unique_degrees), list(counts)

    bins = np.logspace(np.log10(min_deg), np.log10(max_deg), num=num_bins + 1)

    bin_sums_y = np.histogram(unique_degrees, bins=bins, weights=counts)[0]
    sum_of_k_in_bin = np.histogram(unique_degrees, bins=bins, weights=unique_degrees)[0]
    count_of_k_in_bin = np.histogram(unique_degrees, bins=bins)[0]

    valid_bins_mask = count_of_k_in_bin > 0
    bin_means_x = np.divide(sum_of_k_in_bin, count_of_k_in_bin, where=valid_bins_mask)

    plot_x = bin_means_x[valid_bins_mask]
    plot_y = bin_sums_y[valid_bins_mask]

    return list(plot_x), list(plot_y)


def plot_comparisons(
    df,
    metric_name,
    target_model,
    other_models,
    save_path,
    wasserstein_dists=None,
    figsize=(10, 6.5),
    ylabel=None,
):
    """
    Plot comparisons of a specific metric between a target model and other models into a violin plot, faceted by 'top_n' values.
    """
    # Build a long-format DataFrame for Seaborn.
    if ylabel is None:
        ylabel = metric_name.capitalize()
    save_path = Path(save_path)
    df_rows = []
    for top_n_value in df["top_n"].unique():
        if pd.isna(top_n_value):
            df_subset = df[df["top_n"].isna()]
        else:
            df_subset = df[df["top_n"] == top_n_value]

        target_scores = df_subset[df_subset["model"] == target_model][
            metric_name
        ].copy()
        if target_scores.empty:
            continue

        for other_model_name in other_models:
            other_scores = df_subset[df_subset["model"] == other_model_name][
                metric_name
            ].copy()
            if other_scores.empty:
                continue

            comparison_label = f"{other_model_name.upper()}"

            for score in target_scores:
                df_rows.append(
                    {
                        "comparison": comparison_label,
                        "model": "Real",
                        metric_name: score,
                        "run": None,
                        "top_n": top_n_value,
                        "_raw_top_n": top_n_value,
                        "_other_model": other_model_name,
                    }
                )

            other_df = df_subset[df_subset["model"] == other_model_name]
            for _, row in other_df.iterrows():
                df_rows.append(
                    {
                        "comparison": comparison_label,
                        "model": "Generated",
                        metric_name: row[metric_name],
                        "run": row["run"],
                        "top_n": top_n_value,
                        "_raw_top_n": top_n_value,
                        "_other_model": other_model_name,
                    }
                )

    if not df_rows:
        print("No data pairs to plot.")
        return

    long_format_df = pd.DataFrame(df_rows)
    long_format_df["top_n"] = long_format_df["top_n"].fillna("N/A")

    top_n_categories = long_format_df["top_n"].unique()

    print(f"Plotting faceted comparison for {save_path.name}...")

    # num_cols = len(top_n_categories)
    # base_width, base_height = figsize
    # corr_factor = 1.5 if num_cols > 1 else 1.0
    # aspect_ratio = (base_width / num_cols) / base_height * corr_factor

    num_cols = len(top_n_categories)
    base_width, base_height = figsize

    min_facet_width = 8.5

    facet_width = max(
        base_width / num_cols,
        min_facet_width,
    )

    aspect_ratio = facet_width / base_height
    g = sns.FacetGrid(
        long_format_df,
        col="top_n",
        height=base_height,
        aspect=aspect_ratio,
        sharey=True,
        col_order=sorted(top_n_categories, key=lambda x: (isinstance(x, str), x)),
    )

    g.map_dataframe(
        sns.violinplot,
        y=metric_name,
        x="comparison",
        hue="model",
        split=True,
        inner=None,
        linewidth=1.2,
        palette={
            "Real": to_rgba("#4B7CCB", alpha=1.0),
            "Generated": to_rgba("#ED9866", alpha=1.0),
        },
    )

    g.add_legend(
        bbox_to_anchor=(0.5, -0.10),
        loc="lower center",
        ncol=2,
        fontsize=18,  # 14 → 16
        title_fontsize=20,  # 17 → 18
    )
    g.set_axis_labels("", ylabel, fontsize=22)  # 16 → 18
    g.set_xticklabels(rotation=0, ha="center", fontsize=20)  # 14 → 16
    g.set_titles(col_template="n = {col_name}", fontsize=22)  # 16 → 18

    if g._legend:
        plt.setp(g._legend.get_title(), fontsize=20, fontweight="bold")

    g.map_dataframe(
        lambda data, **kwargs: plt.grid(axis="x", linestyle="--", alpha=0.6)
    )

    run_values = sorted(
        long_format_df.loc[long_format_df["model"] == "Generated", "run"]
        .dropna()
        .unique()
    )

    rng = np.random.default_rng(seed=42)

    for i, (ax, col_name) in enumerate(zip(g.axes.flatten(), g.col_names)):
        # scatter

        if col_name == "N/A":
            facet_df = long_format_df[long_format_df["top_n"] == "N/A"]
        else:
            facet_df = long_format_df[long_format_df["top_n"] == col_name]

        comparisons = list(facet_df["comparison"].unique())

        for x, comparison in enumerate(comparisons):
            comp_df = facet_df[facet_df["comparison"] == comparison]

            real_values = comp_df.loc[comp_df["model"] == "Real", metric_name].values

            jitter = rng.uniform(-0.05, 0.05, size=len(real_values))

            ax.scatter(
                x - 0.30 + jitter,
                real_values,
                color="mediumblue",
                alpha=0.2,
                edgecolors="none",
                s=8,
                zorder=3,
                rasterized=True,
            )

            gen_df = comp_df[comp_df["model"] == "Generated"]
            runs = sorted(gen_df["run"].dropna().unique())

            run_offsets = np.linspace(-0.15, 0.15, len(runs))

            for run, run_offset in zip(runs, run_offsets):
                run_df = gen_df[gen_df["run"] == run]
                values = run_df[metric_name].values

                jitter = rng.uniform(-0.03, 0.03, size=len(values))
                ax.scatter(
                    x + 0.33 + jitter + run_offset,
                    values,
                    color="mediumblue",
                    edgecolors="none",
                    alpha=0.2,
                    s=8,
                    zorder=3,
                    rasterized=True,
                )

        if col_name != "N/A":
            percentage = float(col_name) * 100
            display_title = f"d = {col_name} ({percentage:.0f}%)"
        else:
            display_title = ""
        ax.set_title(display_title, fontsize=22, y=1.05)
        ax.tick_params(axis="y", labelsize=18)

        if wasserstein_dists:
            lookup_key_top_n = float(col_name) if col_name != "N/A" else "N/A"

            # Hide "top_n = N/A" label but keep the axis for Wasserstein display
            if lookup_key_top_n == "N/A":
                ax.set_title(
                    "",
                    fontsize=20,
                    y=1.17,  # 16 → 18
                )  # Match the y position of other titles

            for i, comparison in enumerate(comparisons):
                conp_df = facet_df[facet_df["comparison"] == comparison]
                other_model_name = conp_df["_other_model"].iloc[0]

                dist = wasserstein_dists.get((lookup_key_top_n, other_model_name))

                if dist is not None:
                    ax.text(
                        x=i,
                        y=-0.13,
                        # s="WD = " + format_mean_sd(dist[0], dist[1], 2),
                        s=rf"$\text{{WD}}={dist[0]:.2g}\pm{dist[1]:.2g}$",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=18,
                    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close()
    print(f"Saved plot to {save_path}")


def plot_common_edges(adj_target, model_results, n, output_path, row_labels=None):
    """
    Plots the common edges between a target adjacency matrix and multiple model adjacency matrices.
    model_results:
        [
            {'name': 'Model A', 'adj': adj_matrix_a, 'jaccard': 0.45},
            {'name': 'Model B', 'adj': adj_matrix_b, 'jaccard': 0.60},
            {'name': 'Model C', 'adj': adj_matrix_c, 'jaccard': 0.85}
        ]
    """
    colors = ["#000000", "#ff0000", "#1865ff", "#ffffff"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    if isinstance(model_results[0], dict):
        model_results = [model_results]

    num_rows = len(model_results)
    num_cols = len(model_results[0]) + 1  # +1 for the target column

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4 * num_cols, 4 * num_rows),
        facecolor="white",
        gridspec_kw={"wspace": 0.05, "hspace": 0.28},
        squeeze=False,
    )
    # ----------------------------------------
    # column 1: Target
    # ----------------------------------------
    mask_target = get_top_n_edges(adj_target, n, weighted=False)
    mask_target = (mask_target > 0).astype(int)

    target_view = mask_target * 3

    for row_idx, row_results in enumerate(model_results):
        ax_tgt = axes[row_idx, 0]
        ax_tgt.imshow(target_view, cmap=cmap, norm=norm, interpolation="nearest")

        ax_tgt.set_title("Target", fontsize=22, fontweight="bold")
        ax_tgt.axis("off")

        if row_labels is not None:
            ax_tgt.text(
                -0.05,
                0.5,  # Vertically centered just outside the left edge.
                row_labels[row_idx],
                transform=ax_tgt.transAxes,  # Use axes-relative coordinates.
                rotation=90,
                ha="right",
                va="center",
                fontsize=22,
                fontweight="bold",
                color="black",
            )
        elif row_idx == 0:
            ax_tgt.text(
                -0.05,
                0.5,  # Vertically centered just outside the left edge.
                f"n = {n}(Top {int(n * 100)}%)",
                transform=ax_tgt.transAxes,  # Use axes-relative coordinates.
                rotation=90,
                ha="right",
                va="center",
                fontsize=22,
                fontweight="bold",
                color="black",
            )

        for col_idx, item in enumerate(row_results, start=1):
            ax = axes[row_idx, col_idx]
            adj_pred = item["adj"]
            mask_pred = get_top_n_edges(adj_pred, n, weighted=False)
            mask_pred = (mask_pred > 0).astype(int)

            overlay = np.zeros_like(mask_target)
            overlay = np.zeros_like(mask_target)
            overlay[(mask_target == 1) & (mask_pred == 0)] = 1  # Missed (red)
            overlay[(mask_target == 0) & (mask_pred == 1)] = 2  # Noise (blue)
            overlay[(mask_target == 1) & (mask_pred == 1)] = 3  # Common (white)

            ax.imshow(overlay, cmap=cmap, norm=norm, interpolation="nearest")

            ax.set_title(
                f"{item['name']}\nJaccard: {item.get('jaccard', 0):.3f}",
                fontsize=23,
                fontweight="bold",
            )
            ax.axis("off")

    legend_elements = [
        Patch(edgecolor="black", linewidth=1.5, facecolor=colors[3], label="Common"),
        Patch(facecolor=colors[1], label="Target Only"),
        Patch(facecolor=colors[2], label="Generated Only"),
    ]

    # fig.legend(
    #     handles=legend_elements,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, -0.03),
    #     ncol=3,
    #     fontsize=12,
    #     edgecolor="gray",
    # )

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_edge_variability_heatmap(
    matrices: dict[str, np.ndarray],
    output_path: os.PathLike,
    *,
    ncols: int = 3,
    vmin: float | None = None,
    vmax: float | None = None,
):
    arrays = list(matrices.values())

    if vmin is None:
        vmin = min(np.min(arr) for arr in arrays)
    if vmax is None:
        vmax = max(np.max(arr) for arr in arrays)

    n_plots = len(matrices)
    ncols = min(ncols, n_plots)
    nrows = math.ceil(n_plots / ncols)

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_over("red")

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    im = None

    for ax, (label, matrix) in zip(axes.flat, matrices.items()):
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=14)

    for ax in axes.flat[n_plots:]:
        ax.axis("off")

    if im is None:
        raise ValueError("No valid matrices to plot.")

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Edge Variability", extend="max")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved edge variability heatmap to {output_path}")


def plot_edge_variability_scatter(
    gen_var_matrix: np.ndarray,
    ref_var_matrix: np.ndarray,
    output_path: os.PathLike,
    *,
    x_label: str = "Generated Edge Variability (SD)",
    y_label: str = "Reference Edge Variability (SD)",
):
    gen_edges = matrix_to_vec(gen_var_matrix)
    ref_edges = matrix_to_vec(ref_var_matrix)

    max_val = max(np.nanmax(gen_edges), np.nanmax(ref_edges))

    plt.figure(figsize=(6, 6))
    plt.scatter(
        gen_edges,
        ref_edges,
        s=18,
        alpha=0.5,
        color="darkturquoise",
        edgecolor="teal",
        linewidth=0.8,
        rasterized=True,
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    x_lim = max_val * 1.05
    y_lim = max_val * 1.05

    plt.xlim(0, x_lim)
    plt.ylim(0, y_lim)

    plt.plot([0, x_lim], [0, y_lim], "--", linewidth=1, color="black")

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.tick_params(axis="both", labelsize=12)

    r, _ = pearsonr(gen_edges, ref_edges)

    plt.text(
        1.0,
        1.02,
        rf"$r^2 = {r**2:.3f}$",
        transform=plt.gca().transAxes,
        fontsize=16,
        ha="right",
        va="bottom",
        clip_on=False,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close()
    print(f"Saved edge variability scatter plot to {output_path}")


def _add_group_title(fig, axes, title, *, y=0.89, fontsize=18):
    axes = np.asarray(axes)

    left = min(ax.get_position().x0 for ax in axes.flat)
    right = max(ax.get_position().x1 for ax in axes.flat)

    center = (left + right) / 2.0

    fig.text(
        center,
        y,
        title,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
    )


def plot_individual_matrices(
    real,
    vae,
    gan,
    ddpm,
    output_path,
    *,
    weighted_cmap="magma",
    binary_cmap="Greys_r",
    vmin=None,
    vmax=None,
    norm=None,
):
    n_runs = 4
    n_samples = 3

    assert real.shape[0] == n_runs
    assert all(data.shape[:2] == (n_runs, n_samples) for data in [vae, gan, ddpm])

    fig = plt.figure(figsize=(16, 7))

    outer = fig.add_gridspec(
        nrows=1,
        ncols=8,
        width_ratios=[
            1.0,  # Real
            0.25,  # gap
            0.28,  # run labels
            3.0,  # VAE
            0.30,  # gap
            3.0,  # WGAN-GP
            0.30,  # gap
            3.0,  # DDPM
        ],
        left=0.03,
        right=0.99,
        bottom=0.05,
        top=0.88,
        wspace=0.0,
    )

    real_grid = outer[0].subgridspec(
        nrows=n_runs,
        ncols=1,
        hspace=0.05,
    )

    # run labels: 4 rows × 1 col
    label_grid = outer[2].subgridspec(
        nrows=n_runs,
        ncols=1,
        hspace=0.05,
    )

    # Use four rows and three columns per generative model.
    vae_grid = outer[3].subgridspec(
        nrows=n_runs,
        ncols=n_samples,
        wspace=0.03,
        hspace=0.05,
    )

    wgan_grid = outer[5].subgridspec(
        nrows=n_runs,
        ncols=n_samples,
        wspace=0.03,
        hspace=0.05,
    )

    ddpm_grid = outer[7].subgridspec(
        nrows=n_runs,
        ncols=n_samples,
        wspace=0.03,
        hspace=0.05,
    )

    real_axes = []
    vae_axes = np.empty((n_runs, n_samples), dtype=object)
    wgan_axes = np.empty((n_runs, n_samples), dtype=object)
    ddpm_axes = np.empty((n_runs, n_samples), dtype=object)

    is_binary = is_binary_matrix(real)

    if is_binary:
        cmap = plt.get_cmap(binary_cmap)
    else:
        cmap = plt.get_cmap(weighted_cmap)
        cmap.set_over("white")

    # -------------------------
    # Real
    # -------------------------
    for run in range(n_runs):
        ax = fig.add_subplot(real_grid[run, 0])

        im = ax.imshow(real[run], cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)

        ax.set_xticks([])
        ax.set_yticks([])

        real_axes.append(ax)

    # -------------------------
    # run labels
    # -------------------------
    for run in range(n_runs):
        ax = fig.add_subplot(label_grid[run, 0])

        ax.axis("off")

        ax.text(
            0.5,
            0.5,
            f"run {run + 1}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=14,
            fontweight="bold",
        )

    # -------------------------
    # Generated models
    # -------------------------
    model_data = [
        (vae, vae_grid, vae_axes),
        (gan, wgan_grid, wgan_axes),
        (ddpm, ddpm_grid, ddpm_axes),
    ]

    for data, grid, axes in model_data:
        for run in range(n_runs):
            for sample in range(n_samples):
                ax = fig.add_subplot(grid[run, sample])

                ax.imshow(data[run, sample], cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)

                ax.set_xticks([])
                ax.set_yticks([])

                axes[run, sample] = ax

    # -------------------------
    # Model names
    # -------------------------
    # Finalize the axes positions before adding group titles.
    fig.canvas.draw()

    _add_group_title(
        fig,
        real_axes,
        "Real",
    )

    _add_group_title(
        fig,
        vae_axes,
        "VAE",
    )

    _add_group_title(
        fig,
        wgan_axes,
        "WGAN-GP",
    )

    _add_group_title(
        fig,
        ddpm_axes,
        "DDPM",
    )

    if not is_binary:
        right = max(ax.get_position().x1 for ax in ddpm_axes.flat)
        bottom = min(ax.get_position().y0 for ax in ddpm_axes.flat)
        top = max(ax.get_position().y1 for ax in ddpm_axes.flat)

        cax = fig.add_axes(
            (
                float(right + 0.01),
                float(bottom),
                0.012,
                float(top - bottom),
            )
        )

        cbar = fig.colorbar(im, cax=cax, extend="max")
        cbar.set_label("Edge Weight")

    # -------------------------
    # -------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_nn_distance_violin(
    csv_path,
    output_dir,
    dataset,
    models=("vae", "gan", "ddpm"),
    cutoff=None,
    rect_right=None,
):
    df = pd.read_csv(csv_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = df[df["dataset"] == dataset].copy()

    plot_parts = []

    # test
    test_df = dataset_df[dataset_df["source"] == "test"].copy()
    test_df["group"] = "Test"
    plot_parts.append(test_df)

    for model in models:
        model_df = dataset_df[
            (dataset_df["model"] == model) & (dataset_df["source"] == "generated")
        ].copy()

        model_df["group"] = get_formatted_model_name(model)
        plot_parts.append(model_df)

    plot_df = pd.concat(plot_parts, ignore_index=True)

    order = ["Test"] + [get_formatted_model_name(model) for model in models]

    plt.figure(figsize=(10, 6))

    for group in order:
        sns.violinplot(
            data=plot_df[plot_df["group"] == group],
            x="nn_distance",
            y="group",
            order=order,
            orient="h",
            inner="quartile",
            cut=0,
            density_norm="width",
            width=0.5,
            split=True,
            color="#4B7CCB" if group == "Test" else "#ED9866",
        )
    ax = plt.gca()

    if rect_right is not None:
        transform = blended_transform_factory(ax.transData, ax.transAxes)

        rect = Rectangle(
            (0.12, 0.005),
            rect_right - 0.1,
            0.99,
            transform=transform,
            fill=False,
            edgecolor="gray",
            linestyle="--",
            linewidth=2,
            zorder=4,
        )
        ax.add_patch(rect)

    rng = np.random.default_rng(seed=42)

    scatter_base_offset = 0.35
    max_points = 5000

    # plot scatter points for each group with jitter
    for y, group in enumerate(order):
        group_df = plot_df[plot_df["group"] == group]

        if group == "Test":
            values = group_df["nn_distance"].dropna().values
            jitter = rng.uniform(-0.03, 0.03, size=len(values))

            ax.scatter(
                values,
                y + scatter_base_offset + jitter,
                color="mediumblue",
                alpha=0.1,
                edgecolors="none",
                s=6,
                zorder=3,
                rasterized=True,
            )
        else:
            runs = sorted(group_df["run"].dropna().unique())
            run_offsets = np.linspace(0, 0.30, len(runs))

            for run, run_offset in zip(runs, run_offsets):
                run_df = group_df[group_df["run"] == run]
                values = run_df["nn_distance"].dropna().values
                if len(values) > max_points:
                    values = rng.choice(values, size=max_points, replace=False)
                jitter = rng.uniform(-0.03, 0.03, size=len(values))

                ax.scatter(
                    values,
                    y + scatter_base_offset + jitter + run_offset,
                    color="mediumblue",
                    edgecolors="none",
                    alpha=0.1,
                    s=6,
                    zorder=3,
                    rasterized=True,
                )

    if cutoff is None:
        sns.despine()
    plt.xlim(left=0, right=cutoff)

    plt.xlabel("Nearest-neighbor distance to training data", fontsize=16)
    plt.ylabel("")
    plt.title(get_formatted_dataset_name(dataset, abbr=True), fontsize=18)

    plt.tick_params(axis="both", labelsize=16)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{dataset}_nn_distance_violin.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_pairwise_distance_violin(
    csv_path,
    output_dir,
    dataset,
    models=("vae", "gan", "ddpm"),
    cutoff=None,
    rect_right=None,
):
    df = pd.read_csv(csv_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = df[df["Dataset"] == dataset].copy()

    dataset_df["group"] = dataset_df["Model"].map(
        lambda model: "Test" if model == "real" else get_formatted_model_name(model)
    )

    order = ["Test"] + [get_formatted_model_name(model) for model in models]

    plot_df = dataset_df[dataset_df["group"].isin(order)]

    plt.figure(figsize=(10, 6))

    # 9 or 8
    for group in order:
        sns.violinplot(
            data=plot_df[plot_df["group"] == group],
            x="Distance",
            y="group",
            order=order,
            orient="h",
            inner="quartile",
            cut=0,
            density_norm="width",
            width=0.5,
            split=True,
            color="#4B7CCB" if group == "Test" else "#ED9866",
        )
    ax = plt.gca()

    if rect_right is not None:
        transform = blended_transform_factory(ax.transData, ax.transAxes)

        rect = Rectangle(
            (0.12, 0.005),
            rect_right - 0.1,
            0.99,
            transform=transform,
            fill=False,
            edgecolor="gray",
            linestyle="--",
            linewidth=2,
            zorder=4,
        )
        ax.add_patch(rect)

    rng = np.random.default_rng(seed=42)

    scatter_base_offset = 0.35
    max_points = 5000

    # plot scatter points for each group with jitter
    for y, group in enumerate(order):
        group_df = plot_df[plot_df["group"] == group]

        if group == "Test":
            values = group_df["Distance"].dropna().values
            if len(values) > max_points:
                values = rng.choice(values, size=max_points, replace=False)
            jitter = rng.uniform(-0.03, 0.03, size=len(values))

            ax.scatter(
                values,
                y + scatter_base_offset + jitter,
                color="mediumblue",
                alpha=0.1,
                edgecolors="none",
                s=6,
                zorder=3,
                rasterized=True,
            )
        else:
            runs = sorted(group_df["Run"].dropna().unique())
            run_offsets = np.linspace(0, 0.30, len(runs))

            for run, run_offset in zip(runs, run_offsets):
                run_df = group_df[group_df["Run"] == run]
                values = run_df["Distance"].dropna().values
                if len(values) > max_points:
                    values = rng.choice(values, size=max_points, replace=False)
                jitter = rng.uniform(-0.03, 0.03, size=len(values))

                ax.scatter(
                    values,
                    y + scatter_base_offset + jitter + run_offset,
                    color="mediumblue",
                    edgecolors="none",
                    alpha=0.1,
                    s=6,
                    zorder=3,
                    rasterized=True,
                )

    if cutoff is None:
        sns.despine()
    plt.xlim(left=0, right=cutoff)

    plt.xlabel("Pairwise Euclidean distance", fontsize=16)
    plt.ylabel("", fontsize=14)
    plt.title(get_formatted_dataset_name(dataset, abbr=True), fontsize=18)
    plt.tick_params(axis="both", labelsize=16)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{dataset}_pairwise_distance_violin.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
