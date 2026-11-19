import argparse
import os
from multiprocessing import Pool
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from common.graph_util import get_top_n_edges, get_topns
from eval.core.iteration import iter_evaluation_targets
from eval.core.smallworld import calc_single_sigma

DATA_CACHE = {}


def setup_tasks(dataset_name, model_name, run, num_graphs, topns, calculated_set):
    """make tasks list"""
    tasks = []
    for i in range(num_graphs):
        for topn in topns:
            if (dataset_name, model_name, run, i, topn) not in calculated_set:
                tasks.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "run": run,
                        "graph_id": i,
                        "top_n": topn,
                    }
                )
    return tasks


def _get_calculated_set(output_file):
    """Load the set of completed tasks from a CSV file."""
    calculated_set = set()
    if Path(output_file).exists():
        df_done = pd.read_csv(output_file)
        for _, row in df_done.iterrows():
            topn = row.get("top_n", None)
            if pd.isna(topn):
                topn = None

            run = row["run"]
            if pd.isna(run):
                run = None

            calculated_set.add(
                (
                    row["dataset"],
                    row["model"],
                    run,
                    row["graph_id"],
                    topn,
                )
            )
    return calculated_set


def process_task_worker(task):
    try:
        cache_key = (task["dataset"], task["model"], task["run"])
        all_matrixes = DATA_CACHE[cache_key]
        matrix = all_matrixes[task["graph_id"]]
        topn = task["top_n"]
        matrix = get_top_n_edges(matrix, topn)

        sigma, size_ratio = calc_single_sigma(matrix)
        if np.isnan(sigma):
            return (task, None)

        return (task, (sigma, size_ratio))

    except Exception as e:
        print(f"Error processing task {task}: {e}")
        return (task, None)


def main(path_config, dataset):
    # num_graph = 1000
    output_dir = Path(path_config["roots"]["output"]) / "sigma_results"

    output_file = Path(output_dir) / "sigma_results.csv"
    header = [
        "dataset",
        "model",
        "run",
        "graph_id",
        "sigma",
        "size_ratio",
        "top_n",
    ]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not output_file.exists():
        pd.DataFrame(columns=header).to_csv(output_file, index=False)

    print("Pre-loading all .npz files into memory...")
    targets = list(iter_evaluation_targets(path_config, [dataset], include_real=False))

    for target in targets:
        print(
            f"Pre-loading dataset: {target.dataset}, model: {target.model}, run: {target.run}"
        )

        DATA_CACHE[(target.dataset, target.model, target.run)] = target.generated
        DATA_CACHE.setdefault((target.dataset, "real", None), target.ref)

    print("Pre-loading complete.")

    calculated_set = _get_calculated_set(output_file)

    all_tasks = []
    topns = get_topns(dataset)
    for target in targets:
        num_graphs = len(DATA_CACHE[(target.dataset, target.model, target.run)])
        tasks = setup_tasks(
            dataset_name=target.dataset,
            model_name=target.model,
            run=target.run,
            num_graphs=num_graphs,
            topns=topns,
            calculated_set=calculated_set,
        )

        all_tasks.extend(tasks)

    real_key = (dataset, "real", None)
    num_graphs = len(DATA_CACHE[real_key])
    real_tasks = setup_tasks(
        dataset_name=dataset,
        model_name="real",
        run=None,
        num_graphs=num_graphs,
        topns=topns,
        calculated_set=calculated_set,
    )
    all_tasks.extend(real_tasks)

    if not all_tasks:
        print("All tasks have been processed.")
        exit()

    print(f"Total tasks to process: {len(all_tasks)}")

    num_processes = 28
    successful_tasks = 0

    with Pool(num_processes) as pool:
        results_iter = pool.imap_unordered(process_task_worker, all_tasks)

        for task, result in tqdm(results_iter, total=len(all_tasks)):
            if result is not None:
                try:
                    sigma, size_ratio = result
                    row = {
                        "dataset": task["dataset"],
                        "model": task["model"],
                        "run": task["run"],
                        "graph_id": task["graph_id"],
                        "sigma": sigma,
                        "size_ratio": size_ratio,
                        "top_n": task["top_n"],
                    }

                    df_row = pd.DataFrame([row], columns=header)
                    df_row.to_csv(output_file, mode="a", header=False, index=False)
                    successful_tasks += 1
                except Exception as e:
                    print(f"Error writing result for task {task}: {e}")
            else:
                print(f"Failed to process task: {task}")

    print(f"Successfully processed {successful_tasks} out of {len(all_tasks)} tasks.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Calculate sigma for graph datasets."
    )

    argparser.add_argument(
        "--path_config",
        type=str,
        default="config/paths.yaml",
        help="Path to the path_config yaml file.",
    )
    argparser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to evaluate. If omitted, ws, sc, and hcp datasets will be evaluated.",
    )

    args = argparser.parse_args()
    with open(args.path_config, "r") as f:
        import yaml

        path_config = yaml.safe_load(f)

    all_datasets = list(path_config["datasets"].keys())
    if args.datasets is None:
        datasets = ["ws", "nimh", "hcp"]
    else:
        unknown_datasets = set(args.datasets) - set(all_datasets)
        if unknown_datasets:
            raise ValueError(
                f"Unknown datasets specified: {', '.join(unknown_datasets)}. "
                f"Available datasets are: {', '.join(all_datasets)}"
            )

        datasets = args.datasets

    for dataset in datasets:
        main(path_config, dataset)
