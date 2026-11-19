from pathlib import Path

import pandas as pd

from dataset.graph_dataset import get_data_as_numpy
from eval.core.graph_statistics import calc_nearest_neighbor
from eval.core.iteration import iter_evaluation_targets
from eval.core.similaritiy import calc_pairwise_distances


def evaluate_graph_nns(path_config, datasets):
    all_results = []
    evaluated_test_runs = set()

    for target in iter_evaluation_targets(path_config, datasets, include_real=False):
        fold = int(target.run[-1])
        train = get_data_as_numpy(path_config, target.dataset, "train", fold=fold)
        test = target.ref
        gen = target.generated

        # generated -> train
        nn_distances, train_indices = calc_nearest_neighbor(gen, train)

        for gen_idx, (nn_distance, train_idx) in enumerate(
            zip(nn_distances, train_indices)
        ):
            all_results.append(
                {
                    "dataset": target.dataset,
                    "model": target.model,
                    "run": target.run,
                    "source": "generated",
                    "nn_distance": nn_distance,
                    "source_idx": gen_idx,
                    "train_idx": train_idx,
                }
            )

        # test -> train
        # Reuse the result across models for the same dataset and run.
        key = (target.dataset, target.run)

        if key not in evaluated_test_runs:
            nn_distances, train_indices = calc_nearest_neighbor(test, train)

            for test_idx, (nn_distance, train_idx) in enumerate(
                zip(nn_distances, train_indices)
            ):
                all_results.append(
                    {
                        "dataset": target.dataset,
                        "model": "real",
                        "run": target.run,
                        "source": "test",
                        "nn_distance": nn_distance,
                        "source_idx": test_idx,
                        "train_idx": train_idx,
                    }
                )

            evaluated_test_runs.add(key)

        print(
            f"Evaluated nearest neighbor for dataset: {target.dataset}, "
            f"model: {target.model}, run: {target.run}"
        )

    df = pd.DataFrame(all_results)

    output_path = Path(path_config["roots"]["output"]) / "nearest_neighbor_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Nearest neighbor results saved to: {output_path}")


def evaluate_pairwise_distances(path_config, datasets):
    all_results = []

    for target in iter_evaluation_targets(path_config, datasets, include_real=True):
        samples = target.ref if target.model == "real" else target.generated
        pairwise_distances = calc_pairwise_distances(samples)

        for i, j, distance in pairwise_distances:
            all_results.append(
                {
                    "Dataset": target.dataset,
                    "Model": target.model,
                    "Run": target.run,
                    "Sample_i": i,
                    "Sample_j": j,
                    "Distance": distance,
                }
            )

    df = pd.DataFrame(all_results)
    output_path = (
        Path(path_config["roots"]["output"])
        / "metrics"
        / "pairwise_distances_results.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Pairwise distances results saved to: {output_path}")
