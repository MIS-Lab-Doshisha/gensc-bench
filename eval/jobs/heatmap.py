import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common.visualize import (
    format_mean_sd,
    get_formatted_dataset_name,
    get_formatted_metric_name,
    get_formatted_model_name,
)

HIGHER_IS_BETTER_METRICS = {"Planarity"}


def get_metric_columns(df):
    metadata_columns = {"Dataset", "Run", "Top n Edges", "Model"}
    return [col for col in df.columns if col not in metadata_columns]


def normalize_by_metric(df):
    normalized = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for metric in df.columns:
        column = df[metric]
        min_val = column.min()
        max_val = column.max()

        if pd.isna(min_val) or pd.isna(max_val):
            normalized[metric] = np.nan
        elif min_val == max_val:
            normalized[metric] = 0.5
        else:
            normalized[metric] = (column - min_val) / (max_val - min_val)

            if metric in HIGHER_IS_BETTER_METRICS:
                normalized[metric] = 1.0 - normalized[metric]

    return normalized


def plot_heatmap_panels(
    plot_data,
    annot_data,
    *,
    split_by,
    output_path,
    ylabel,
    row_label_formatter=None,
    row_height=2.4,
    width=15.0,
):
    font_size_annot = 12
    font_size_tick = 14
    font_size_text = 14

    if split_by not in plot_data.index.names:
        raise ValueError(f"split_by '{split_by}' not in DataFrame index levels.")

    split_values = plot_data.index.get_level_values(split_by).unique()
    normalized_data = normalize_by_metric(plot_data)

    current_cmap = plt.get_cmap("magma_r").copy()
    current_cmap.set_under("white")

    row_counts = (
        plot_data.groupby(level=split_by, sort=False).size().reindex(split_values)
    )

    fig_height = max(2.0, row_height * len(split_values))

    fig, axes = plt.subplots(
        nrows=len(split_values),
        ncols=1,
        figsize=(width, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": row_counts.values},
    )

    if len(split_values) == 1:
        axes = [axes]

    for ax, split_value in zip(axes, split_values):
        data_subset = normalized_data.xs(split_value, level=split_by).fillna(-1.0)
        annot_subset = annot_data.xs(split_value, level=split_by)

        sns.heatmap(
            data_subset,
            ax=ax,
            fmt="",
            annot=annot_subset,
            linewidths=0.5,
            linecolor="gray",
            cmap=current_cmap,
            vmin=0.0,
            vmax=1.0,
            cbar=False,
            annot_kws={"size": font_size_annot},
        )

        ax.tick_params(axis="y", labelrotation=0)

        ax.set_xlabel("")
        ax.tick_params(
            axis="x", bottom=False, labelbottom=False, top=False, labeltop=False
        )

        if split_by == "Dataset":
            split_label = get_formatted_dataset_name(split_value, abbr=True)
        else:
            split_label = "d=" + str(split_value)

        ax.text(
            -0.12,
            0.5,
            split_label,
            transform=ax.transAxes,
            rotation=90,
            va="center",
            fontsize=font_size_text,
            fontweight="bold",
        )

        ax.tick_params(axis="y", labelsize=font_size_tick)
        ax.set_ylabel(ylabel, fontsize=font_size_text)

    top_ax = axes[0]
    top_ax.xaxis.set_label_position("top")
    top_ax.set_xlabel(
        "Metrics", fontsize=font_size_text, fontweight="bold", labelpad=12
    )
    top_ax.tick_params(axis="x", labeltop=True, top=False, labelsize=font_size_tick)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {output_path}")


def plot_mean_sd_heatmap(df, *, split_by, output_path, split_order=None, width=15.0):
    df.columns = df.columns.map(get_formatted_metric_name)
    metric_columns = get_metric_columns(df)
    df["Model"] = df["Model"].str.lower().map(get_formatted_model_name)

    model_order = ["Real", "VAE", "WGAN-GP", "DDPM"]

    df["Model"] = pd.Categorical(
        df["Model"],
        categories=model_order,
        ordered=True,
    )

    if split_order is not None:
        df[split_by] = pd.Categorical(
            df[split_by],
            categories=split_order,
            ordered=True,
        )

    group_columns = [split_by, "Model"]
    grouped = df.groupby(
        group_columns,
        sort=True,
        dropna=False,
        observed=True,
    )[metric_columns]

    mean_df = grouped.mean()
    sd_df = grouped.std(ddof=1)
    count_df = grouped.count()

    annot_df = pd.DataFrame(index=mean_df.index, columns=mean_df.columns, dtype=object)
    for metric in mean_df.columns:
        annot_df[metric] = [
            format_mean_sd(mean, sd, count)
            for mean, sd, count in zip(mean_df[metric], sd_df[metric], count_df[metric])
        ]

    plot_heatmap_panels(
        mean_df,
        annot_df,
        split_by=split_by,
        output_path=output_path,
        ylabel="",
        width=width,
    )
