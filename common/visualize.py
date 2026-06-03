import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from common.graph_util import get_thresholded_matrix
from common.util import get_reconstruction_from_npz


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


def _is_binary_matrix(matrix):
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
    cmap_heat="gist_heat",
    cmap_binary="gray",
    global_min=None,
    global_max=None,
    column_title_kwards={"fontsize": 16},
    title_kwards={"fontsize": 20},
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
        if not _is_binary_matrix(matrix):
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
                vmin=global_min,
                vmax=global_max,
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
    fig.colorbar(mappable, cax=cax, **cbar_kwards)

    # plt.tight_layout()
    output_dir = Path(output_path_name).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_path_name).with_suffix(".pdf"), bbox_inches="tight")
    plt.show()


def plot_multiple_thresh(matrices, thresh_list, output_path_name="None"):
    size = matrices[0].shape[0]
    all_matrices_list = [matrices]
    for thresh in thresh_list:
        binarized_matrices = get_thresholded_matrix(matrices, threshold=thresh)
        all_matrices_list.append(binarized_matrices)

    stacked_matrices = np.stack(all_matrices_list, axis=1)
    all_matrices = stacked_matrices.reshape(-1, size, size)

    name = ["Original"] + [f"Threshold {thresh}" for thresh in thresh_list]
    plot_matrix_grid(
        all_matrices,
        output_path_name=output_path_name,
        grid_size=(len(matrices), len(thresh_list) + 1),
        column_titles=name,
    )


def plot_all_with_multiple_thresh():
    with open("path_list.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    del data["wba"]

    generated_path = [
        path for item in data.values() for path in item["reconstructions"]
    ]
    thresh = [0.05, 0.1, 0.2, 0.3]
    for path in generated_path:
        recon = get_reconstruction_from_npz(path)[:5]
        plot_multiple_thresh(
            recon, thresh, output_path_name=Path(path).parent / "recon_thresh_mmd"
        )


def plot_weighted_graphs_grid(graphs: list, nrows: int, ncols: int, path: str = None):
    """
    Plot weighted graphs in a grid layout.

    Args:
        graphs (list): List of networkx.Graph objects or adjacency matrices.
        nrows (int): Number of rows in the grid.
        ncols (int): Number of columns in the grid.
        path (str, optional): Path to save the figure.
    """
    graphs = [nx.from_numpy_array(matrix) for matrix in graphs]
    num_graphs = len(graphs)
    if num_graphs == 0:
        print("No graphs to plot.")
        return

    fig, axes = plt.subplots(
        nrows,
        ncols + 1,
        figsize=(6 * (ncols + 0.5), 6 * nrows),
        gridspec_kw={"width_ratios": [1] * ncols + [0.1]},
    )

    if nrows == 1:
        if ncols == 1:
            axes = [axes]
        graph_axes = axes[:ncols]
        colorbar_ax = axes[ncols]
    else:
        graph_axes = axes[:, :ncols].flatten()
        colorbar_ax = axes[0, -1]
        for r in range(1, nrows):
            axes[r, -1].axis("off")

    # Collect all edge weights for consistent color scaling
    all_weights = [
        data["weight"]
        for g in graphs
        if g.number_of_edges() > 0
        for u, v, data in g.edges(data=True)
    ]

    vmin, vmax, norm, cmap, sm = (None,) * 5
    if all_weights:
        vmin = min(all_weights)
        vmax = max(all_weights)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.Greys
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    else:
        print(
            "Warning: No edge weights found in graphs. Colorbar will not be displayed."
        )

    # Draw each graph
    for i, g in enumerate(graphs):
        ax = graph_axes[i]

        if g.number_of_edges() > 0:
            edges, weights = zip(*nx.get_edge_attributes(g, "weight").items())
            pos = nx.spring_layout(g, seed=i)

            nx.draw(
                g,
                pos,
                ax=ax,
                with_labels=False,
                edge_color=weights,
                edge_cmap=cmap,
                edge_vmin=vmin,
                edge_vmax=vmax,
                edgelist=edges,
                node_size=200,
                node_color="darkturquoise",
                width=1.5,
            )
        else:
            pos = nx.spring_layout(g, seed=i)
            nx.draw(
                g, pos, ax=ax, with_labels=False, node_size=200, node_color="lightblue"
            )

        ax.set_title(f"Graph {i + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()

    for j in range(num_graphs, len(graph_axes)):
        graph_axes[j].axis("off")

    # Add colorbar
    if all_weights:
        fig.colorbar(sm, cax=colorbar_ax, label="Edge Weight")
    else:
        colorbar_ax.axis("off")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    if path:
        plt.savefig(path, bbox_inches="tight")
        print(f"Graph grid saved to {path}")

    plt.show()

    plt.clf()
    plt.close()


def radar_plot(labels, data, data_labels, save_path=None, reverse=True):
    length = len(labels)

    angle_list = [n / float(length) * 2 * np.pi for n in range(length)]
    angle_list += angle_list[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    data = np.array(data, dtype=np.float32)
    max_val_par_feature = np.max(data, axis=0)
    min_val_par_feature = np.min(data, axis=0)
    ranges = max_val_par_feature - min_val_par_feature
    ranges[ranges == 0] = 1.0
    data = (data - min_val_par_feature) / ranges
    if reverse:
        data = 1 - data

    for i, d in enumerate(data):
        d_closed = np.append(d, d[0])
        ax.plot(angle_list, d_closed, label=data_labels[i])
        ax.fill(angle_list, d_closed, alpha=0.1)

    ax.set_thetagrids(np.degrees(angle_list[:-1]), labels, fontsize=12)
    ax.set_ylim(-0.2, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.2"], fontsize=10)
    plt.legend(loc="upper right")

    if save_path:
        plt.savefig(save_path)


def radar_plot_from_csv_with_hue(csv_path, hue_column=None):
    """
    Create radar charts from CSV data, optionally grouped by hue_column.
    If hue_column is None, the entire file is plotted as a single chart.
    """
    file_path = Path(csv_path)
    plot_labels = ["Deg.", "Clus.", "Orbit", "Spec.", "Emb."]

    try:
        df = pd.read_csv(file_path, index_col=0, header=0)
        print(f"\nProcessing file: {file_path}")
        name = file_path.stem.split("_")[0]

        # Remove "Validity" column for planar graphs if it exists
        if name == "planar" and "Validity" in df.columns:
            df = df.drop("Validity", axis=1)

        # Check that all required plot labels exist in the DataFrame
        if not all(label in df.columns for label in plot_labels):
            raise ValueError(
                f"One or more required plot labels {plot_labels} not found in DataFrame columns."
            )

        # If hue_column is specified, group and plot
        if hue_column is not None:
            if hue_column not in df.columns:
                raise ValueError(
                    f"hue_column '{hue_column}' not found in DataFrame columns."
                )

            for hue_value, group_df in df.groupby(hue_column):
                print(f"  -> Plotting group: {hue_value}")

                data = group_df[plot_labels].values
                data_labels = group_df["Model"].tolist()

                safe_hue_value = str(hue_value).replace(".", "_").replace(" ", "_")
                save_path = f"output/plot/{name}_{safe_hue_value}_radar.pdf"

                radar_plot(plot_labels, data, data_labels, save_path=save_path)

        # If hue_column is None, plot the entire file
        else:
            print("  -> Plotting entire file (no hue)")

            data = df[plot_labels].values
            data_labels = df["Model"].tolist()

            save_path = f"output/plot/{name}_radar.pdf"

            radar_plot(plot_labels, data, data_labels, save_path=save_path)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


def get_formatted_dataset_name(key: str, abbr=False) -> str:
    if abbr:
        name_mapping = {
            "ba": "BA",
            "planar": "Planar",
            "sbm": "SBM",
            "wba": "WSF",
            "ws": "WS",
            "sc": "SC",
        }
    else:
        name_mapping = {
            "ba": "Barabási-Albert",
            "planar": "Planar",
            "sbm": "Stochastic Block Model",
            "wba": "Weighted Scale-Free",
            "ws": "Watts-Strogatz",
            "sc": "Structural Connectivity",
        }
    return name_mapping.get(key, key)
