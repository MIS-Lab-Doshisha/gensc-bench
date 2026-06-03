import os
from multiprocessing import Pool
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.graph_util import get_top_n_edges, get_topns
from common.path_manager import PathManager
from common.util import get_reconstruction_from_npz
from dataset.graph_dataset import get_testdata_as_numpy
from eval.eval_func import calc_single_sigma

DATA_CACHE = {}  # グローバルキャッシュ辞書


def setup_tasks(dataset_name, model_name, num_graphs, topns, calculated_set):
    """make tasks list"""
    tasks = []
    for i in range(num_graphs):
        for topn in topns:
            if (dataset_name, model_name, i, topn) not in calculated_set:
                tasks.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "graph_id": i,
                        "topn": topn,
                    }
                )
    return tasks


def _get_calculated_set(output_file):

    calculated_set = set()
    if Path(output_file).exists():
        df_done = pd.read_csv(output_file)
        for _, row in df_done.iterrows():
            topn = row.get("topn", None)
            if pd.isna(topn):
                topn = None

            calculated_set.add(
                (row["dataset_name"], row["model_name"], row["graph_id"], topn)
            )
    return calculated_set


def process_task_worker(task):
    try:
        cache_key = (task["dataset"], task["model"])
        all_matrixes = DATA_CACHE[cache_key]
        matrix = all_matrixes[task["graph_id"]]
        topn = task["topn"]
        matrix = get_top_n_edges(matrix, topn)

        sigma, size_ratio = calc_single_sigma(matrix)
        if np.isnan(sigma):
            return (task, None)

        return (task, (sigma, size_ratio))

    except Exception as e:
        print(f"Error processing task {task}: {e}")
        return (task, None)


if __name__ == "__main__":
    datasets = ["sc"]
    models = ["real"]
    # num_graph = 1000
    output_dir = "output/metrics"

    output_file = Path(output_dir) / "sigma_results.csv"
    header = ["dataset_name", "model_name", "graph_id", "sigma", "size_ratio", "topn"]

    # --- 1. Prepare CSV header first ---
    if not output_file.exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=header).to_csv(output_file, index=False)

    # --- 2. Pre-load all data in the parent process once ---
    print("Pre-loading all .npz files into memory...")
    manager = PathManager("processed_path_list.json")
    for dataset in datasets:
        for model in models:
            if (dataset, model) in DATA_CACHE:
                continue

            if model == "original":
                graph = get_testdata_as_numpy(dataset)
                DATA_CACHE[(dataset, model)] = graph
            else:
                graph_path = manager.get_recon_path(dataset, model)
                DATA_CACHE[(dataset, model)] = get_reconstruction_from_npz(graph_path)
    print("Pre-loading complete.")

    # --- 3. Load already processed tasks once ---
    calculated_set = _get_calculated_set(output_file)

    # --- 4. Create task list per graph ---
    all_tasks = []
    for dataset in datasets:
        topns = get_topns(dataset)
        for model in models:
            num_graph = len(DATA_CACHE[(dataset, model)])
            tasks = setup_tasks(  # Call the modified setup_tasks
                dataset_name=dataset,
                model_name=model,
                num_graphs=num_graph,
                topns=topns,
                calculated_set=calculated_set,  # Pass the loaded set
            )
            all_tasks.extend(tasks)

    if not all_tasks:
        print("All tasks have been processed.")
        exit()

    print(f"Total tasks to process: {len(all_tasks)}")  # Results in 6000 tasks

    num_processes = 28
    successful_tasks = 0

    with Pool(num_processes) as pool:
        results_iter = pool.imap_unordered(process_task_worker, all_tasks)

        for task, result in tqdm(results_iter, total=len(all_tasks)):
            if result is not None:
                try:
                    sigma, size_ratio = result
                    row = {
                        "dataset_name": task["dataset"],
                        "model_name": task["model"],
                        "graph_id": task["graph_id"],
                        "sigma": sigma,
                        "size_ratio": size_ratio,
                        "topn": task["topn"],
                    }

                    df_row = pd.DataFrame([row], columns=header)
                    df_row.to_csv(output_file, mode="a", header=False, index=False)
                    successful_tasks += 1
                except Exception as e:
                    print(f"Error writing result for task {task}: {e}")
            else:
                print(f"Failed to process task: {task}")

    print(f"Successfully processed {successful_tasks} out of {len(all_tasks)} tasks.")
