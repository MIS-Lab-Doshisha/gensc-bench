import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.path_manager import PathManager
from common.util import get_reconstruction_from_npz
from common.visualize import (
    get_formatted_dataset_name,
    plot_matrix_grid,
    radar_plot_from_csv_with_hue,
)
from dataset.graph_dataset import get_testdata_as_numpy
from eval.eval_func import calc_wasserstein_distance
from eval.plot import plot_comparisons, plot_heatmap_from_df


def plot_radar_from_csv():
    paths = Path("output/mmd_results").rglob("*_mmds.csv")

    for path in paths:
        name = path.stem.split("_")[0]
        radar_plot_from_csv_with_hue(
            csv_path=path, hue_column="Top N Edges" if name == "sc" else None
        )


def plot_modularities_from_csv():
    path = Path("output/metrics/modularity_scores.csv")
    df_all = pd.read_csv(path)
    datasets = df_all["dataset"].unique()
    target_model = "real"
    other_models = ["vae", "wgan-gp", "ddpm"]

    for dataset in datasets:
        df = df_all[df_all["dataset"] == dataset]

        wd_dists = {}

        unique_top_ns = df["top_n"].unique()

        for top_n_val in unique_top_ns:
            if pd.isna(top_n_val):
                sub_df = df[df["top_n"].isna()]
            else:
                sub_df = df[df["top_n"] == top_n_val]

            # ターゲット（基準）のスコアを取得
            target_scores = sub_df[sub_df["model"] == target_model]["modularity"].values

            if len(target_scores) == 0:
                print(f"Warning: No target data for top_n={top_n_val}")
                continue

            # 各比較モデルに対して距離を計算
            for other_model in other_models:
                other_scores = sub_df[sub_df["model"] == other_model][
                    "modularity"
                ].values

                if len(other_scores) == 0:
                    continue

                # 距離計算
                wd = calc_wasserstein_distance(target_scores, other_scores)
                key_top_n = "N/A" if pd.isna(top_n_val) else top_n_val

                # 辞書に格納
                wd_dists[(key_top_n, other_model)] = wd

                print(
                    f"Computed WD [top_n={top_n_val}]: {target_model} vs {other_model} = {wd:.4f}"
                )

        plot_comparisons(
            df,
            metric_name="modularity",
            target_model="real",
            other_models=["vae", "wgan-gp", "ddpm"],
            save_path=f"output/plots/modularity/{dataset}_modularity_wd_comparisons.pdf",
            wasserstein_dists=wd_dists,
        )


def plot_sigmas_from_csv():
    path = Path("output/metrics/sigma_results.csv")
    df_all = pd.read_csv(path)
    datasets = df_all["dataset"].unique()
    target_model = "real"
    other_models = ["vae", "wgan-gp", "ddpm"]

    for dataset in datasets:
        df = df_all[df_all["dataset"] == dataset]

        wd_dists = {}

        unique_top_ns = df["top_n"].unique()

        for top_n_val in unique_top_ns:
            if pd.isna(top_n_val):
                sub_df = df[df["top_n"].isna()]
            else:
                sub_df = df[df["top_n"] == top_n_val]

            # ターゲット（基準）のスコアを取得
            target_scores = sub_df[sub_df["model"] == target_model]["sigma"].values

            if len(target_scores) == 0:
                print(f"Warning: No target data for top_n={top_n_val}")
                continue

            # 各比較モデルに対して距離を計算
            for other_model in other_models:
                other_scores = sub_df[sub_df["model"] == other_model]["sigma"].values

                if len(other_scores) == 0:
                    continue

                # 距離計算
                wd = calc_wasserstein_distance(target_scores, other_scores)
                key_top_n = "N/A" if pd.isna(top_n_val) else top_n_val

                # 辞書に格納
                wd_dists[(key_top_n, other_model)] = wd

                print(
                    f"Computed WD [top_n={top_n_val}]: {target_model} vs {other_model} = {wd:.4f}"
                )

        plot_comparisons(
            df,
            metric_name="sigma",
            target_model="real",
            other_models=["vae", "wgan-gp", "ddpm"],
            save_path=f"output/plots/sigma/{dataset}_sigma_wd_comparisons.pdf",
            wasserstein_dists=wd_dists,
        )


def plot_processed_graph():
    keys = ["ba", "planar", "sbm", "wba", "ws"]
    paths = []

    for key in keys:
        paths.extend(Path(f"output/processed/{key}").rglob("*.npz"))

    for path in paths:
        data = get_reconstruction_from_npz(path)
        plot_matrix_grid(data[:9], output_path_name=path.with_suffix(".pdf"))


def plot_heatmaps():
    # plot synthetic datasets
    keys = ["ba", "planar", "sbm", "ws", "wba"]
    dataframs = []

    for key in keys:
        path = Path(f"output/mmd_results/{key}_mmds.csv")

        df = pd.read_csv(path, index_col=0)
        df["Dataset"] = key
        dataframs.append(df)

    combined_df = pd.concat(dataframs, ignore_index=True)
    combined_df.set_index(["Dataset", "Model"], inplace=True)

    plot_heatmap_from_df(
        combined_df, output_path="output/mmd_results/syn_mmd_heatmap.pdf"
    )

    # plot sc dataset
    key = "sc"
    path = Path(f"output/mmd_results/{key}_mmds.csv")

    df = pd.read_csv(path, index_col=0)
    df.set_index(["Top N", "Model"], inplace=True)

    plot_heatmap_from_df(
        df, output_path="output/mmd_results/sc_mmd_heatmap.pdf", split_by="Top N"
    )


def plot_graphs_summary():
    manager = PathManager("processed_path_list.json")
    output_dir = Path("output/raw_vis")
    keys = manager.get_all_keys()

    for key in keys:
        # --- Load original test data ---
        orig_data = get_testdata_as_numpy(key)

        # --- Load generated data for each model ---
        models = ["vae", "gan", "ddpm"]

        gen_data_per_model = {}
        for model in models:
            gen_path = manager.get_recon_path(key, model)
            gen_data_per_model[model] = get_reconstruction_from_npz(gen_path)

        # --- Random 3 samples from original ---
        orig_idx = np.random.choice(len(orig_data), size=3, replace=False)
        orig_samples = orig_data[orig_idx]

        # --- Random 3 samples from each model ---
        gen_samples = {
            model: gen_data_per_model[model][
                np.random.choice(len(gen_data_per_model[model]), size=3, replace=False)
            ]
            for model in models
        }

        # --- Build matrices in correct order ---
        matrices = []
        for i in range(3):  # row = 3 samples
            matrices.append(orig_samples[i])  # left-most col = original
            for model in models:
                matrices.append(gen_samples[model][i])

        # --- Column titles ---
        column_titles = ["Original"] + [m.upper() for m in models]

        # --- Output path ---
        output_path_name = str(Path(output_dir) / f"{key}_comparison")
        dataset_name = get_formatted_dataset_name(key)

        if key == "wba":
            vmax = 2.0
        elif key == "sc":
            vmax = 37.0
        else:
            vmax = None

        # --- Call your helper function ---
        plot_matrix_grid(
            matrices=matrices,
            output_path_name=output_path_name,
            grid_size=(3, 1 + len(models)),
            column_titles=column_titles,
            plot_title=f"{dataset_name} Samples",
            cmap_heat="gist_heat",
            cmap_binary="gray",
            global_max=vmax,
            column_title_kwards={"fontsize": 24},
            title_kwards={"fontsize": 28, "fontweight": "bold"},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modularity", action="store_true", help="Plot modularity comparisons"
    )
    parser.add_argument(
        "--radar", action="store_true", help="Plot radar charts from CSVs"
    )
    parser.add_argument("--sigma", action="store_true", help="Plot sigma comparisons")
    parser.add_argument(
        "--plot_graphs", action="store_true", help="Plot processed graphs"
    )
    parser.add_argument(
        "--heatmaps", action="store_true", help="Plot heatmaps from MMD results"
    )
    args = parser.parse_args()

    if args.modularity:
        plot_modularities_from_csv()
    if args.radar:
        plot_radar_from_csv()
    if args.sigma:
        plot_sigmas_from_csv()
    if args.plot_graphs:
        # plot_processed_graph()
        plot_graphs_summary()
    if args.heatmaps:
        plot_heatmaps()
