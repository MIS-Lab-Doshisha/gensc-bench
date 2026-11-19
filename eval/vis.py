import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import PowerNorm
from scipy.stats import wasserstein_distance

from common.util import get_reconstruction_from_npz
from common.visualize import (
    get_formatted_model_name,
    plot_matrix_grid,
)
from eval.core.graph_statistics import calc_edge_variability
from eval.core.iteration import iter_evaluation_targets
from eval.jobs.heatmap import plot_mean_sd_heatmap
from eval.jobs.plot import (
    _get_common_scale_symmetric,
    plot_comparisons,
    plot_edge_variability_heatmap,
    plot_edge_variability_scatter,
    plot_individual_matrices,
    plot_mean_matrices,
    plot_nn_distance_violin,
    plot_pairwise_distance_violin,
)


def plot_modularities_from_csv(path_config, datasets):
    path = Path("output/metrics/modularity_scores.csv")
    df_all = pd.read_csv(path)
    top_ns = [0.05, 0.15, 0.3]
    wd_records = []

    target_model = "real"
    other_models = path_config["models"]

    for dataset in datasets:
        df = df_all[df_all["dataset"] == dataset].copy()

        wd_dists = {}

        if df["top_n"].notna().any():
            df = df[df["top_n"].isin(top_ns)].copy()
            unique_top_ns = top_ns
        else:
            unique_top_ns = [np.nan]

        for top_n_val in unique_top_ns:
            if pd.isna(top_n_val):
                sub_df = df[df["top_n"].isna()]
            else:
                sub_df = df[df["top_n"] == top_n_val]

            target_scores = sub_df[sub_df["model"] == target_model]["modularity"].values

            if len(target_scores) == 0:
                print(f"Warning: No target data for top_n={top_n_val}")
                continue

            for other_model in other_models:
                model_df = sub_df[sub_df["model"] == other_model]

                if len(model_df) == 0:
                    continue

                run_wds = []

                for run, run_df in model_df.groupby("run"):
                    other_scores = run_df["modularity"].values
                    if len(other_scores) == 0:
                        continue

                    wd = wasserstein_distance(target_scores, other_scores)
                    run_wds.append(wd)

                    wd_records.append(
                        {
                            "dataset": dataset,
                            "top_n": top_n_val,
                            "target_model": target_model,
                            "other_model": other_model,
                            "run": run,
                            "wasserstein_distance": wd,
                        }
                    )

                    print(
                        f"Computed WD [top_n={top_n_val}, run={run}]: "
                        f"{target_model} vs {other_model} = {wd:.4f}"
                    )

                if len(run_wds) == 0:
                    continue

                wd_mean = np.mean(run_wds)
                wd_sd = np.std(run_wds) if len(run_wds) > 1 else 0.0

                key_top_n = "N/A" if pd.isna(top_n_val) else top_n_val

                wd_dists[(key_top_n, get_formatted_model_name(other_model))] = (
                    wd_mean,
                    wd_sd,
                )

                print(
                    f"WD summary [top_n={top_n_val}]: {target_model} vs {other_model} = {wd_mean:.4f} ± {wd_sd:.4f}"
                )

        df["model"] = df["model"].apply(get_formatted_model_name)

        plot_comparisons(
            df,
            metric_name="modularity",
            target_model="Real",
            other_models=[get_formatted_model_name(m) for m in other_models],
            save_path=f"output/metrics/modularity/{dataset}_modularity_wd_comparisons.pdf",
            wasserstein_dists=wd_dists,
            figsize=(8.5, 6.5),
        )
        pd.DataFrame(wd_records).to_csv(
            "output/metrics/modularity/modularity_wd_records.csv", index=False
        )


def plot_sigmas_from_csv(path_config, datasets):
    path = Path("output/sigma_results/sigma_results.csv")
    df_all = pd.read_csv(path)
    top_ns = [0.05, 0.15, 0.3]
    sigma_records = []

    target_model = "real"
    other_models = path_config["models"]

    for dataset in datasets:
        df = df_all[df_all["dataset"] == dataset].copy()

        wd_dists = {}

        if df["top_n"].notna().any():
            df = df[df["top_n"].isin(top_ns)].copy()
            unique_top_ns = top_ns
        else:
            unique_top_ns = [np.nan]

        for top_n_val in unique_top_ns:
            if pd.isna(top_n_val):
                sub_df = df[df["top_n"].isna()]
            else:
                sub_df = df[df["top_n"] == top_n_val]

            target_scores = sub_df[sub_df["model"] == target_model]["sigma"].values

            if len(target_scores) == 0:
                print(f"Warning: No target data for top_n={top_n_val}")
                continue

            for other_model in other_models:
                model_df = sub_df[sub_df["model"] == other_model]

                if len(model_df) == 0:
                    continue

                raw_wds = []

                for run, run_df in model_df.groupby("run"):
                    other_scores = run_df["sigma"].values
                    if len(other_scores) == 0:
                        continue

                    wd = wasserstein_distance(target_scores, other_scores)
                    raw_wds.append(wd)
                    sigma_records.append(
                        {
                            "dataset": dataset,
                            "top_n": top_n_val,
                            "target_model": target_model,
                            "other_model": other_model,
                            "run": run,
                            "wasserstein_distance": wd,
                        }
                    )
                    print(
                        f"Computed WD [top_n={top_n_val}, run={run}]: "
                        f"{target_model} vs {other_model} = {wd:.4f}"
                    )

                if len(raw_wds) == 0:
                    continue

                wd_mean = np.mean(raw_wds)
                wd_sd = np.std(raw_wds) if len(raw_wds) > 1 else 0.0

                key_top_n = "N/A" if pd.isna(top_n_val) else top_n_val

                wd_dists[(key_top_n, get_formatted_model_name(other_model))] = (
                    wd_mean,
                    wd_sd,
                )

                print(
                    f"WD summary [top_n={top_n_val}]: "
                    f"{target_model} vs {other_model} = {wd_mean:.4f} ± {wd_sd:.4f}"
                )

        df["model"] = df["model"].apply(get_formatted_model_name)

        plot_comparisons(
            df,
            metric_name="sigma",
            target_model="Real",
            other_models=[get_formatted_model_name(m) for m in other_models],
            save_path=f"output/metrics/sigma/{dataset}_sigma_wd_comparisons.pdf",
            wasserstein_dists=wd_dists,
            figsize=(8.5, 6.5),
            ylabel="Small-worldness",
        )
        pd.DataFrame(sigma_records).to_csv(
            "output/metrics/sigma/sigma_wd_records.csv", index=False
        )


def plot_processed_graph():
    keys = ["ba", "planar", "sbm", "wba", "ws"]
    paths = []

    for key in keys:
        paths.extend(Path(f"output/processed/{key}").rglob("*.npz"))

    for path in paths:
        data = get_reconstruction_from_npz(path)
        plot_matrix_grid(data[:9], output_path_name=path.with_suffix(".pdf"))


def plot_heatmaps(path_config, datasets):
    # plot synthetic datasets
    dataframs = []
    top_n_edges = [0.05, 0.15, 0.3]

    for dataset in datasets:
        if not path_config["datasets"][dataset]["type"] == "synthetic":
            continue
        path = Path(f"output/mmd_results/{dataset}_mmds.csv")

        df = pd.read_csv(path, index_col=0)
        df["Dataset"] = dataset
        dataframs.append(df)

    combined_df = pd.concat(dataframs, ignore_index=True)

    plot_mean_sd_heatmap(
        combined_df,
        split_by="Dataset",
        split_order=datasets,
        output_path="output/mmd_results/syn_mmd_heatmap.pdf",
    )

    # plot sc dataset
    for key in ["nimh", "hcp"]:
        order = [
            "Model",
            "Run",
            "Top n Edges",
            "Degree MMD",
            "Clustering MMD",
            "Orbit MMD",
            "Spectral MMD",
            "Embedding MMD",
            "Strength MMD",
        ]
        path = Path(f"output/mmd_results/{key}_mmds.csv")

        df = pd.read_csv(path, index_col=None)
        df = df[order]

        if top_n_edges is not None:
            df = df[df["Top n Edges"].isin(top_n_edges)]

        df = df.sort_values("Top n Edges")

        plot_mean_sd_heatmap(
            df,
            output_path=f"output/mmd_results/{key}_mmd_heatmap.pdf",
            split_by="Top n Edges",
            width=12.5,
        )


def plot_individual_graphs(path_config, datasets):
    output_dir = Path(f"{path_config['roots']['output']}/individual_graphs")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=42)

    for dataset in datasets:
        real = None
        vae = []
        gan = []
        ddpm = []

        for target in iter_evaluation_targets(
            path_config, [dataset], include_real=False
        ):
            if real is None:
                real = target.ref

            if target.model == "vae":
                vae.append(target.generated)
            elif target.model == "gan":
                gan.append(target.generated)
            elif target.model == "ddpm":
                ddpm.append(target.generated)

        assert real is not None

        real_indices = rng.choice(len(real), size=4, replace=False)
        real_samples = real[real_indices]

        vae_samples = np.stack(
            [data[rng.choice(len(data), size=3, replace=False)] for data in vae]
        )

        gan_samples = np.stack(
            [data[rng.choice(len(data), size=3, replace=False)] for data in gan]
        )

        ddpm_samples = np.stack(
            [data[rng.choice(len(data), size=3, replace=False)] for data in ddpm]
        )

        vmin, vmax, norm = None, None, None

        if path_config["datasets"][dataset]["weighted"]:
            vmin = 0.0
            vmax = float(
                np.percentile(
                    np.concatenate(
                        [
                            real_samples.ravel(),
                            vae_samples.ravel(),
                            gan_samples.ravel(),
                            ddpm_samples.ravel(),
                        ]
                    ),
                    99.9,
                )
            )

            if path_config["datasets"][dataset]["type"] == "sc":
                norm = PowerNorm(gamma=0.3, vmin=0.0, vmax=vmax)
                vmin, vmax = None, None

        plot_individual_matrices(
            real_samples,
            vae_samples,
            gan_samples,
            ddpm_samples,
            output_path=output_dir / f"{dataset}_individual_graphs.pdf",
            vmin=vmin,
            vmax=vmax,
            norm=norm,
        )


def eval_mean(path_config, datasets):
    dataset_data = {}

    for dataset in datasets:
        test_data = None
        data_by_model = {}

        for target in iter_evaluation_targets(
            path_config, [dataset], include_real=False
        ):
            if test_data is None:
                test_data = target.ref

            data_by_model.setdefault(target.model, []).append(
                (target.run, target.generated)
            )

        # plot average martices for all models across runs
        data_dict = {}
        for model, generated_list in data_by_model.items():
            if model == "original":
                mean_data = generated_list[0]
            else:
                mean_data = np.mean(
                    np.stack([gen for _, gen in generated_list], axis=0), axis=0
                )

            data_dict[get_formatted_model_name(model)] = mean_data

        dataset_data[dataset] = {
            "test_data": test_data,
            "data_dict": data_dict,
            "data_by_model": data_by_model,
        }

    all_diffs = {}

    for dataset, data in dataset_data.items():
        target_mean = np.mean(data["test_data"], axis=0)
        for model, matrices in data["data_dict"].items():
            diff = target_mean - matrices
            all_diffs[f"{dataset}_{model}"] = diff

    global_diff_vmin, global_diff_vmax = _get_common_scale_symmetric(all_diffs)
    # with open("output/mean_matrices/nimh_diff_vmin_vmax.pkl", "rb") as f:
    #     global_diff_vmin, global_diff_vmax = pickle.load(f)
    # global_diff_vmin, global_diff_vmax = -0.5, 0.5

    for dataset, data in dataset_data.items():
        norm = None
        mean_ticks = None
        diff_ticks = None

        test_data = data["test_data"]
        data_dict = data["data_dict"]
        data_by_model = data["data_by_model"]

        if path_config["datasets"][dataset]["type"] == "sc":
            norm = PowerNorm(gamma=0.3)

        plot_mean_matrices(
            test_data,
            data_dict,
            output_path=f"output/mean_matrices/{dataset}_mean_orig.pdf",
            cmap_mean="magma",
            norm=norm,
            cbar_ticks_mean=mean_ticks,
            cbar_ticks_diff=diff_ticks,
            diff_vmin=global_diff_vmin,
            diff_vmax=global_diff_vmax,
        )

        # bplot mean matrices for each runs
        for model, generated_list in data_by_model.items():
            run_groups = {f"{run}": generated for run, generated in generated_list}

            plot_mean_matrices(
                test_data,
                run_groups,
                output_path=f"output/mean_matrices/{dataset}/{model}_per_run.pdf",
                cmap_mean="viridis",
                norm=norm,
                cbar_ticks_mean=mean_ticks,
                cbar_ticks_diff=diff_ticks,
            )


def plot_edge_variability(path_config, datasets):
    output_dir = Path(f"{path_config['roots']['output']}/edge_variability")
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        original_variability = None
        model_variabilities = defaultdict(list)

        for target in iter_evaluation_targets(path_config, [dataset]):
            if original_variability is None:
                original_variability = calc_edge_variability(target.ref)

            gen_data = target.generated

            if target.model == "real":
                continue

            variability = calc_edge_variability(gen_data)
            model_variabilities[target.model].append((target.run, variability))

        model_mean_variabilities = {
            model: np.mean([var for _, var in run_variabilities], axis=0)
            for model, run_variabilities in model_variabilities.items()
        }

        assert original_variability is not None, (
            "Original variability should not be None"
        )

        all_matrices = [original_variability]
        for run_variabilities in model_variabilities.values():
            for _, variability in run_variabilities:
                all_matrices.append(variability)
        all_matrices.extend(model_mean_variabilities.values())

        all_matrices = np.concatenate([matrix.ravel() for matrix in all_matrices])
        vmax = float(np.percentile(all_matrices, 99.5))

        for model, run_variabilities in model_variabilities.items():
            matrices = {str(run): variability for run, variability in run_variabilities}
            matrices["Mean"] = model_mean_variabilities[model]

            plot_edge_variability_heatmap(
                matrices=matrices,
                output_path=dataset_dir / f"{model}_edge_variability.pdf",
                ncols=4,
                vmin=0.0,
                vmax=vmax,
            )

        summary_matrices = {
            "Real": original_variability,
            **{
                get_formatted_model_name(model): mean_variability
                for model, mean_variability in model_mean_variabilities.items()
            },
        }

        plot_edge_variability_heatmap(
            matrices=summary_matrices,
            output_path=dataset_dir / "summary_edge_variability.pdf",
            ncols=4,
            vmin=0.0,
            vmax=vmax,
        )

        for model, mean_variability in model_mean_variabilities.items():
            plot_edge_variability_scatter(
                mean_variability,
                original_variability,
                output_path=dataset_dir
                / f"{model}_edge_variability_scatter_rasterized.pdf",
            )


def plot_nn_distance(datasets):
    cutoffs = [220, 14]
    for dataset, cutoff in zip(datasets, cutoffs):
        plot_nn_distance_violin(
            csv_path="output/metrics/nearest_neighbor_results.csv",
            output_dir="output/metrics/nn/",
            dataset=dataset,
            rect_right=cutoff,
        )
        plot_nn_distance_violin(
            csv_path="output/metrics/nearest_neighbor_results.csv",
            output_dir="output/metrics/nn/ext",
            dataset=dataset,
            cutoff=cutoff,
        )


def plot_pairwise_distances(datasets):
    cutoffs = [150, 9]
    for dataset, cutoff in zip(datasets, cutoffs):
        plot_pairwise_distance_violin(
            csv_path="output/metrics/pairwise_distances_results.csv",
            output_dir="output/metrics/pdist/all",
            dataset=dataset,
            rect_right=cutoff,
        )
        plot_pairwise_distance_violin(
            csv_path="output/metrics/pairwise_distances_results.csv",
            output_dir="output/metrics/pdist/ext",
            dataset=dataset,
            cutoff=cutoff,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_config",
        default="config/paths.yaml",
        help="Path to the path configuration file.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to visualize (default: all)",
    )
    parser.add_argument(
        "--modularity", action="store_true", help="Plot modularity comparisons"
    )
    parser.add_argument("--mean", action="store_true", help="Plot mean matrices")
    parser.add_argument("--sigma", action="store_true", help="Plot sigma comparisons")
    parser.add_argument(
        "--plot_graphs", action="store_true", help="Plot processed graphs"
    )
    parser.add_argument(
        "--heatmaps", action="store_true", help="Plot heatmaps from MMD results"
    )
    parser.add_argument(
        "--nn", action="store_true", help="Plot nearest neighbor distance violin plots"
    )
    parser.add_argument(
        "--pdist", action="store_true", help="Plot pairwise distance violin plots"
    )
    parser.add_argument("--edge_var", action="store_true", help="Plot edge variability")
    args = parser.parse_args()

    with open(args.path_config, "r") as f:
        path_config = yaml.safe_load(f)

    all_datasets = path_config["datasets"].keys()
    if args.datasets is None:
        datasets = all_datasets
    elif args.datasets == ["syn"]:
        datasets = ["ba", "planar", "sbm", "ws", "wba"]
    else:
        unknown_datasets = set(args.datasets) - set(all_datasets)
        if unknown_datasets:
            raise ValueError(
                f"Unknown datasets specified: {', '.join(unknown_datasets)}. "
                f"Available datasets are: {', '.join(all_datasets)}"
            )

        datasets = args.datasets

    if args.modularity:
        plot_modularities_from_csv(path_config, datasets)
    if args.sigma:
        plot_sigmas_from_csv(path_config, datasets)
    if args.mean:
        eval_mean(path_config, datasets)
    if args.plot_graphs:
        plot_individual_graphs(path_config, datasets)
    if args.heatmaps:
        plot_heatmaps(path_config, datasets)
    if args.nn:
        plot_nn_distance(datasets)
    if args.edge_var:
        plot_edge_variability(path_config, datasets)
    if args.pdist:
        plot_pairwise_distances(datasets)
