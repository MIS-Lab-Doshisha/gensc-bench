import collections
import math
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from common.graph_util import get_top_n_edges
from common.visualize import get_formatted_dataset_name


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
    conparison_groups,
    output_path,
    cmap_mean="gist_heat",
    cmap_diff="seismic",
):
    target_mean = np.mean(target_matrices, axis=0)

    comp_means = {
        label: np.mean(matrices, axis=0)
        for label, matrices in conparison_groups.items()
    }
    diff_means = {
        label: target_mean - comp_mean for label, comp_mean in comp_means.items()
    }

    vmin, vmax = _get_common_scale(target_mean, comp_means)
    diff_vmin, diff_vmax = _get_common_scale_symmetric(diff_means)

    num_cols = len(comp_means) + 1

    fig, axes = plt.subplots(
        2, num_cols, figsize=(4 * num_cols, 8), squeeze=False, constrained_layout=True
    )
    cmap_mean = plt.get_cmap(cmap_mean)
    cmap_mean.set_over("cyan")
    cmap_diff = plt.get_cmap(cmap_diff)
    cmap_diff.set_over("yellow")
    cmap_diff.set_under("green")

    ax = axes[0, 0]
    im_mean = ax.imshow(target_mean, cmap=cmap_mean, vmin=vmin, vmax=vmax)
    ax.set_title("Target Mean Matrix", fontsize=16, y=-0.1)
    ax.set_aspect("equal")
    ax.axis("off")

    for i, (label, comp_mean) in enumerate(comp_means.items()):
        ax = axes[0, i + 1]
        im_comp = ax.imshow(comp_mean, cmap=cmap_mean, vmin=vmin, vmax=vmax)
        ax.set_title(f"{label} Mean Matrix", fontsize=16, y=-0.1)
        ax.set_aspect("equal")
        ax.axis("off")

    axes[1, 0].axis("off")

    for i, (label, diff_mean) in enumerate(diff_means.items()):
        ax = axes[1, i + 1]
        im_diff = ax.imshow(diff_mean, cmap=cmap_diff, vmin=diff_vmin, vmax=diff_vmax)
        ax.set_title(f"Target - {label}", fontsize=16, y=-0.1)
        ax.set_aspect("equal")
        ax.axis("off")

    cbar_mean = fig.colorbar(
        im_mean, ax=axes[0, :], fraction=0.05, pad=0.01, extend="max"
    )
    cbar_mean.set_label("Mean Value", rotation=270, labelpad=15, fontsize=14)
    cbar_diff = fig.colorbar(
        im_diff, ax=axes[1, :], fraction=0.05, pad=0.01, extend="both"
    )
    cbar_diff.set_label("Difference", rotation=270, labelpad=15, fontsize=14)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)
    plt.savefig(output_path, bbox_inches="tight")
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


def plot_degree_distribution_grid(
    matrices,
    filename,
    nrows=None,
    ncols=None,
    log_scale=True,
    log_binning=False,
    num_bins=20,
):
    """
    複数の隣接行列から次数分布の散布図をプロットする（y軸はノード数）。
    """
    num_matrices = len(matrices)
    if num_matrices == 0:
        print("プロットする行列がありません。")
        return

    if nrows is None and ncols is None:
        ncols = math.ceil(math.sqrt(num_matrices))
        nrows = math.ceil(num_matrices / ncols)
    elif nrows is None:
        nrows = math.ceil(num_matrices / ncols)
    elif ncols is None:
        ncols = math.ceil(num_matrices / nrows)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
    )
    axes = axes.flatten()

    for i, matrix in enumerate(matrices):
        ax = axes[i]
        G = nx.from_numpy_array(matrix)
        degrees = [d for n, d in G.degree()]

        if degrees:
            x, y = get_degree_distribution_points(degrees, log_binning, num_bins)
            if x and y:
                ax.scatter(x, y, marker="o", c="navy", s=50)
        else:
            ax.text(0.5, 0.5, "No nodes", ha="center", va="center")

        ax.set_title(f"Matrix {i + 1} Degree Distribution")
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("Degree (k)")
        ax.set_ylabel("Number of Nodes N(k)")
        ax.grid(True, which="both", ls="--", alpha=0.5)

    for i in range(num_matrices, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=3.0)
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")
    plt.close()
    print(f"saved degreee distribution grid to {filename}")


def plot_degree_hist_grid(matrices, filename, nrows=None, ncols=None):
    """
    複数の隣接行列を受け取り、次数分布をグリッド状に並べてプロットし、1つのファイルに保存する。

    Args:
        matrices (list of np.ndarray): 隣接行列のリスト。
        filename (str): 保存する画像ファイル名。
        nrows (int, optional): グリッドの行数。指定しない場合は自動計算。
        ncols (int, optional): グリッドの列数。指定しない場合は自動計算。
    """
    num_matrices = len(matrices)
    if num_matrices == 0:
        print("プロットする行列がありません。")
        return

    if nrows is None and ncols is None:
        ncols = math.ceil(math.sqrt(num_matrices))
        nrows = math.ceil(num_matrices / ncols)
    elif nrows is None:
        nrows = math.ceil(num_matrices / ncols)
    elif ncols is None:
        ncols = math.ceil(num_matrices / nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if num_matrices > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, matrix in enumerate(matrices):
        ax = axes[i]
        G = nx.from_numpy_array(matrix)
        degrees = [d for n, d in G.degree()]

        if degrees:
            sns.histplot(degrees, discrete=True, ax=ax, color="navy", shrink=0.8)
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(left=0, right=xmax * 1.05)

            locator = mticker.MaxNLocator(nbins=10, integer=True)
            ax.xaxis.set_major_locator(locator)
        else:
            ax.text(0.5, 0.5, "No nodes", ha="center", va="center")
            ax.set_xlim(left=0)  # ノードがない場合もx軸の左端を0に

        ax.set_title(f"Matrix {i + 1} Degree Distribution")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    # 使われなかった余分な描画領域を非表示にする
    for i in range(num_matrices, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=3.0)  # タイトルとラベルが重ならないようにパディングを調整
    # PDF形式で保存
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")
    plt.close()
    print(f"saved degreee distribution grid to {filename}")


def plot_comparisons(
    df,
    metric_name,
    target_model,
    other_models,
    save_path,
    wasserstein_dists=None,
    figsize=(10, 6.5),
):
    # --- 1. Seaborn用のロングフォーマットDFを動的に組み立てる ---
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
                        "top_n": top_n_value,
                        "_raw_top_n": top_n_value,
                        "_other_model": other_model_name,
                    }
                )

            for score in other_scores:
                df_rows.append(
                    {
                        "comparison": comparison_label,
                        "model": "Generated",
                        metric_name: score,
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

    num_cols = len(top_n_categories)
    base_width, base_height = figsize
    corr_factor = 1.5 if num_cols > 1 else 1.0
    aspect_ratio = (base_width / num_cols) / base_height * corr_factor

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
        palette={"Real": "#4C72B0", "Generated": "#DD8452"},
    )

    g.map_dataframe(
        sns.stripplot,
        y=metric_name,
        x="comparison",
        hue="model",
        dodge=True,
        palette={"Real": "#717171", "Generated": "#717171"},
        alpha=0.5,
        size=3,
        legend=False,
    )

    g.add_legend(
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=18,  # 14 → 16
        title_fontsize=20,  # 17 → 18
    )
    g.set_axis_labels("Compared Pair", metric_name.capitalize(), fontsize=22)  # 16 → 18
    g.set_xticklabels(rotation=45, ha="right", fontsize=20)  # 14 → 16
    g.set_titles(col_template="n = {col_name}", fontsize=22)  # 16 → 18

    if g._legend:
        # 凡例のタイトル ("Model") のサイズ
        plt.setp(g._legend.get_title(), fontsize=20, fontweight="bold")

    g.map_dataframe(
        lambda data, **kwargs: plt.grid(axis="x", linestyle="--", alpha=0.6)
    )

    for ax, col_name in zip(g.axes.flatten(), g.col_names):
        if col_name != "N/A":
            percentage = float(col_name) * 100
            display_title = f"n = {col_name} ({percentage:.0f}%)"
        else:
            display_title = ""
        ax.set_title(display_title, fontsize=22, y=1.22)
        ax.tick_params(axis="y", labelsize=18)

        if wasserstein_dists:
            lookup_key_top_n = col_name
            if col_name != "N/A":
                lookup_key_top_n = float(col_name)

            y_max = long_format_df[metric_name].max()
            y_min = long_format_df[metric_name].min()
            y_range = y_max - y_min

            text_y_pos = y_max + 0.28 * y_range  # Move WD label higher up
            plt.ylim(top=y_max + 0.25 * y_range)  # Increase top margin for WD labels

            # Hide "top_n = N/A" label but keep the axis for Wasserstein display
            if lookup_key_top_n == "N/A":
                ax.set_title(
                    "",
                    fontsize=20,
                    y=1.17,  # 16 → 18
                )  # Match the y position of other titles

            ax.text(
                x=0.5,
                y=1.21,
                s="Wasserstein Distance",
                transform=ax.transAxes,  # 軸に対する相対座標(0.0~1.0)
                ha="center",
                va="top",
                fontsize=20,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
            )
            xticklabels = [label.get_text() for label in ax.get_xticklabels()]

            for i, label_text in enumerate(xticklabels):
                try:
                    other_model_name = label_text.lower()
                except IndexError:
                    continue  # ラベル形式が想定と違う場合はスキップ

                dist = wasserstein_dists.get((lookup_key_top_n, other_model_name))

                if dist is not None:
                    ax.text(
                        x=i,
                        y=text_y_pos,
                        s=f"{dist:#.2g}".rstrip("."),
                        ha="center",
                        va="bottom",
                        fontsize=20,  # 14 → 16
                        color="black",
                    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved plot to {save_path}")


def prepare_comparison_data(ref_mats, gen_mats_dict, metric_name, calc_metric_func):
    """
    参照と生成データの行列リストから、比較用DataFrameとWDスコアを準備する。

    Args:
        ref_mats (list): 参照データの行列 (np.ndarray) のリスト。
        gen_mats_dict (dict): 生成データの辞書。
                              {'NAME_A': [mat1, mat2, ...], 'NAME_B': ...} の形式。
        metric_name (str): 計算する指標の名前 (DataFrameの列名になる)。
        calc_metric_func (function): 行列リストから指標値のリストを計算する関数。

    Returns:
        tuple: (pd.DataFrame, dict) - プロット用のDataFrameとWDスコアの辞書
    """
    data_frames_list = []
    wd_scores = {}

    print(f"Calculating {metric_name} for: Reference")
    ref_values = list(calc_metric_func(ref_mats))

    for name, gen_mats in gen_mats_dict.items():
        name = name.upper()
        print(f"Calculating {metric_name} for: {name}")

        generated_values = list(calc_metric_func(gen_mats))

        # calc wasserstein distance
        wd = (ref_values, generated_values)
        comparison_name = f"Reference vs {name}"
        wd_scores[comparison_name] = wd

        # dataframe for plot
        df = pd.DataFrame(
            {
                metric_name: ref_values + generated_values,
                "group": ["Reference"] * len(ref_values)
                + ["Generated"] * len(generated_values),
                "comparison": comparison_name,
            }
        )
        data_frames_list.append(df)

    return pd.concat(data_frames_list, ignore_index=True), wd_scores


def plot_common_edges(adj_target, model_results, n, output_path):
    """
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

    num_cols = len(model_results) + 1
    fig, axes = plt.subplots(
        1,
        num_cols,
        figsize=(4 * num_cols, 4),
        facecolor="white",
        gridspec_kw={"wspace": 0.05},
    )
    # ----------------------------------------
    # column 1: Target
    # ----------------------------------------
    ax_tgt = axes[0]
    mask_target = get_top_n_edges(adj_target, n, weighted=False)
    mask_target = (mask_target > 0).astype(int)

    target_view = mask_target * 3

    ax_tgt.imshow(target_view, cmap=cmap, norm=norm, interpolation="nearest")

    ax_tgt.text(
        -0.05,
        0.5,  # 座標 (左外側, 中央)
        f"n = {n}(Top {int(n * 100)}%)",
        transform=ax_tgt.transAxes,  # 軸を基準にした相対座標
        rotation=90,  # 横書き
        ha="right",  # 右揃え（グラフに寄せる）
        va="center",  # 上下中央揃え
        fontsize=20,
        fontweight="bold",
        color="black",  # 明示的に黒にする
    )
    ax_tgt.set_title("Target", fontsize=20, fontweight="bold")
    ax_tgt.axis("off")

    # ----------------------------------------
    # columns 2~: Models
    # ----------------------------------------
    for i, item in enumerate(model_results):
        ax = axes[i + 1]
        name = item["name"]
        adj_pred = item["adj"]
        jaccard = item.get("jaccard", 0)

        mask_pred = get_top_n_edges(adj_pred, n, weighted=False)
        mask_pred = (mask_pred > 0).astype(int)

        # Overlay logic
        overlay = np.zeros_like(mask_target)
        overlay[(mask_target == 1) & (mask_pred == 0)] = 1  # Missed (赤)
        overlay[(mask_target == 0) & (mask_pred == 1)] = 2  # Noise (青)
        overlay[(mask_target == 1) & (mask_pred == 1)] = 3  # Common (白)

        ax.imshow(overlay, cmap=cmap, norm=norm, interpolation="nearest")

        # Include Jaccard in the title
        ax.set_title(
            f"{name}\nJaccard: {jaccard:.3f}",
            fontsize=18,
            fontweight="bold",
        )
        ax.axis("off")

    legend_elements = [
        Patch(edgecolor="black", linewidth=1.5, facecolor=colors[3], label="Common"),
        Patch(facecolor=colors[1], label="Target Only"),
        Patch(facecolor=colors[2], label="Generated Only"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        fontsize=12,
        edgecolor="gray",
    )

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_heatmap_from_df(df, output_path, split_by="Dataset"):
    font_size_annot = 12
    font_size_tick = 12
    font_size_text = 14
    higher_is_better_metrics = ["Val."]

    if split_by not in df.index.names:
        raise ValueError(f"split_by '{split_by}' not in DataFrame index levels.")

    split_values = df.index.get_level_values(split_by).unique()

    global_min = df.min()
    global_max = df.max()

    current_cmap = plt.get_cmap("magma_r").copy()
    current_cmap.set_under("white")

    fig, axes = plt.subplots(
        nrows=len(split_values),
        ncols=1,
        figsize=(8, 2 * len(split_values)),
        sharex=True,
    )

    if len(split_values) == 1:
        axes = [axes]

    for i, val in enumerate(split_values):
        data_subset = df.xs(val, level=split_by)

        annot_labels = data_subset.map(lambda x: f"{x:#.2g}" if not pd.isna(x) else "-")
        plot_data = (data_subset - global_min) / (global_max - global_min)
        for metric in higher_is_better_metrics:
            if metric in plot_data.columns:
                plot_data[metric] = 1.0 - plot_data[metric]
        plot_data = plot_data.fillna(-1.0)

        sns.heatmap(
            plot_data,
            ax=axes[i],
            fmt="",
            annot=annot_labels,
            linewidths=0.5,
            linecolor="gray",
            cmap=current_cmap,
            vmin=0.0,
            vmax=1.0,
            cbar=False,
            annot_kws={"size": font_size_annot},
        )
        axes[i].set_xlabel("")
        axes[i].tick_params(
            axis="x", bottom=False, labelbottom=False, top=False, labeltop=False
        )
        axes[i].text(
            -0.18,
            0.5,
            get_formatted_dataset_name(split_values[i], abbr=True),
            transform=axes[i].transAxes,
            rotation=90,
            va="center",
            fontsize=font_size_text,
            fontweight="bold",
        )

        axes[i].tick_params(axis="y", labelsize=font_size_tick)
        axes[i].set_ylabel("Models", fontsize=font_size_text)

    top_ax = axes[0]

    top_ax.xaxis.set_label_position("top")
    top_ax.set_xlabel("Metrics", fontsize=font_size_text, fontweight="bold")

    top_ax.tick_params(axis="x", labeltop=True, top=False, labelsize=font_size_tick)

    # cbar = fig.colorbar(
    #     axes[0].collections[0],
    #     ax=axes,
    #     fraction=0.03,
    #     orientation="horizontal",
    #     pad=0.25,
    # )
    # cbar.ax.tick_params(labelsize=font_size_tick)
    # cbar.set_label("Score", fontsize=font_size_text)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {output_path}")
