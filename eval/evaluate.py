import argparse
from pathlib import Path

import pandas as pd
import yaml

from common.graph_util import get_topn_adj_list
from common.util import (
    setup_logger,
)
from eval.core.graph_mmd import strength_stats
from eval.core.graph_statistics import get_planarity_rate
from eval.core.iteration import iter_evaluation_targets
from eval.jobs.eval_jobs import evaluate_graphs
from eval.jobs.metrics_jobs import (
    compute_modularities,
    get_and_plot_jaccard,
    get_strength_spearman,
)
from eval.jobs.nearest_neighbor import evaluate_graph_nns, evaluate_pairwise_distances
from eval.jobs.threshold import save_all_thresholded_matrices


def eval_mmds(path_config, datasets):
    ouput_dir = Path(path_config["roots"]["output"]) / "mmd_results"
    ouput_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=f"{ouput_dir}/eval_mmds.log")

    for dataset in datasets:
        csv_path = f"{path_config['roots']['output']}/mmd_results/{dataset}_mmds.csv"
        ref = None

        results = []

        for target in iter_evaluation_targets(path_config, [dataset]):
            if ref is None:
                ref = target.ref

            logger.info(
                f"Evaluating dataset: {target.dataset}, model: {target.model}, run: {target.run}"
            )

            if not path_config["datasets"][dataset]["thresholding"]:
                top_n_edges = [0.05, 0.1, 0.15, 0.2, 0.3]
            else:
                top_n_edges = [None]

            generated = target.generated

            for top_n in top_n_edges:
                mmds = evaluate_graphs(ref, generated, logger, top_n_edges=top_n)
                row_data = {"Model": target.model.upper(), "Run": target.run}

                if top_n is not None:
                    row_data["Top n Edges"] = top_n

                row_data.update(mmds)
                results.append(row_data)

        df = pd.DataFrame(results)
        columns = [
            "Model",
            "Run",
        ]
        if not path_config["datasets"][dataset]["thresholding"]:
            columns.append("Top n Edges")

        columns += [
            "Degree MMD",
            "Clustering MMD",
            "Orbit MMD",
            "Spectral MMD",
        ]
        df = df.reindex(columns=columns)

        df.to_csv(csv_path, mode="w")
        print(f"Saved evaluation results to {csv_path}")


def eval_strength_mmd(path_config, datasets):
    for dataset in datasets:
        use_topn = not path_config["datasets"][dataset]["thresholding"]
        csv_path = f"{path_config['roots']['output']}/mmd_results/{dataset}_mmds.csv"
        df = pd.read_csv(csv_path)
        all_data = []
        test_data = None

        for target in iter_evaluation_targets(path_config, [dataset]):
            if test_data is None:
                test_data = target.ref

            if use_topn:
                topns = [0.05, 0.1, 0.15, 0.2, 0.3]
                for top_n in topns:
                    test_data_thresh = get_topn_adj_list(test_data, top_n=top_n)
                    recon_data_thresh = get_topn_adj_list(target.generated, top_n=top_n)
                    mmds = strength_stats(test_data_thresh, recon_data_thresh)
                    all_data.append(
                        {
                            "Model": target.model.upper(),
                            "Run": target.run,
                            "Top n Edges": top_n,
                            "Strength MMD": mmds,
                        }
                    )
                    print(
                        f"Dataset: {dataset}, Model: {target.model}, Run: {target.run}, Top n Edges: {top_n}, Strength MMD: {mmds}"
                    )
            else:
                mmds = strength_stats(test_data, target.generated)
                all_data.append(
                    {
                        "Model": target.model.upper(),
                        "Run": target.run,
                        "Top n Edges": None,
                        "Strength MMD": mmds,
                    }
                )
                print(
                    f"Dataset: {dataset}, Model: {target.model}, Run: {target.run}, Strength MMD: {mmds}"
                )

        strength_df = pd.DataFrame(all_data)
        if use_topn:
            merge_keys = ["Model", "Top n Edges", "Run"]
        else:
            merge_keys = ["Model", "Run"]

        try:
            df = df.merge(strength_df, how="left", on=merge_keys)
        except Exception as e:
            print(f"Error merging strength MMDs for dataset {dataset}: {e}")
            continue

        df.to_csv(f"{csv_path}", mode="w", index=False)
        print(f"Saved updated MMD results with Strength MMD to {csv_path}")


def eval_planarity(path_config, datasets):
    for dataset in datasets:
        csv_path = f"{path_config['roots']['output']}/mmd_results/{dataset}_mmds.csv"

        df = pd.read_csv(csv_path)
        all_data = []

        for target in iter_evaluation_targets(path_config, [dataset]):
            print(
                f"Evaluating dataset: {target.dataset}, model: {target.model}, run: {target.run}"
            )

            planarity = get_planarity_rate(target.generated)

            all_data.append(
                {
                    "Model": target.model.upper(),
                    "Run": target.run,
                    "Planarity Rate": planarity,
                }
            )

            print(
                f"Dataset: {dataset}, Model: {target.model}, Run: {target.run}, "
                f"Planarity Rate: {planarity:.4f}"
            )

        planarity_df = pd.DataFrame(all_data)

        try:
            df = df.merge(planarity_df, how="left", on=["Model", "Run"])
        except Exception as e:
            print(f"Error merging planarity rates for dataset {dataset}: {e}")
            continue

        df.to_csv(f"{csv_path}", mode="w", index=False)
        print(f"Saved updated MMD results with Planarity Rate to {csv_path}")


def thresh(path_config, datasets):
    save_all_thresholded_matrices(path_config, datasets)


def nearest_neighbor(path_config, datasets):
    evaluate_graph_nns(path_config, datasets)


def pairwise_distances(path_config, datasets):
    evaluate_pairwise_distances(path_config, datasets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_config",
        default="config/paths.yaml",
        help="Path to the path configuration YAML file",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to evaluate. If omitted, all datasets will be evaluated.",
    )
    parser.add_argument(
        "--mmds", action="store_true", help="Evaluate MMDs for generated graphs"
    )
    parser.add_argument(
        "--thresh", action="store_true", help="Threshold generated graphs"
    )
    parser.add_argument(
        "--strength", action="store_true", help="Evaluate Strength MMDs"
    )
    parser.add_argument("--modularity", action="store_true", help="Evaluate Modularity")
    parser.add_argument(
        "--planarity", action="store_true", help="Evaluate Planarity Rates"
    )
    parser.add_argument(
        "--hub", action="store_true", help="Evaluate Hub Spearman Correlation"
    )
    parser.add_argument(
        "--jaccard", action="store_true", help="Evaluate Jaccard Similarity"
    )
    parser.add_argument(
        "--nn", action="store_true", help="Evaluate Nearest Neighbor Distances"
    )
    parser.add_argument(
        "--pdist", action="store_true", help="Evaluate Pairwise Distances"
    )
    args = parser.parse_args()

    with open(args.path_config, "r") as f:
        path_config = yaml.safe_load(f)

    all_datasets = path_config["datasets"].keys()
    if args.datasets is None:
        datasets = all_datasets
    else:
        unknown_datasets = set(args.datasets) - set(all_datasets)
        if unknown_datasets:
            raise ValueError(
                f"Unknown datasets specified: {', '.join(unknown_datasets)}. "
                f"Available datasets are: {', '.join(all_datasets)}"
            )

        datasets = args.datasets

    args = parser.parse_args()
    if args.mmds:
        eval_mmds(path_config, datasets)
    if args.thresh:
        thresh(path_config, datasets)
    if args.strength:
        eval_strength_mmd(path_config, datasets)
    if args.modularity:
        compute_modularities(path_config, datasets)
    if args.planarity:
        eval_planarity(path_config, datasets)
    if args.hub:
        get_strength_spearman(path_config, datasets)
    if args.jaccard:
        get_and_plot_jaccard(path_config, datasets)
    if args.nn:
        nearest_neighbor(path_config, datasets)
    if args.pdist:
        pairwise_distances(path_config, datasets)
