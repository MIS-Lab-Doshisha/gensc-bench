from dataclasses import dataclass
from typing import Optional

import numpy as np

from common.path_resolver import get_graph_path
from common.util import get_reconstruction_from_npz
from dataset.graph_dataset import get_data_as_numpy


@dataclass
class EvaluationTarget:
    dataset: str
    model: str
    run: Optional[str]
    ref: np.ndarray
    generated: np.ndarray


def iter_evaluation_targets(path_config, datasets, include_real=True):
    for dataset in datasets:
        test_data = get_data_as_numpy(path_config, dataset, split="test")

        if include_real:
            discovery_data = get_data_as_numpy(path_config, dataset, split="discovery")

            yield EvaluationTarget(
                dataset=dataset,
                model="real",
                run=None,
                ref=test_data,
                generated=discovery_data,
            )

        for model in path_config["models"]:
            for run in path_config["runs"].keys():
                path = get_graph_path(path_config, dataset, model, run)
                generated = get_reconstruction_from_npz(path)

                yield EvaluationTarget(
                    dataset=dataset,
                    model=model,
                    run=run,
                    ref=test_data,
                    generated=generated,
                )


if __name__ == "__main__":
    import yaml

    config_path = "config/paths.yaml"
    with open(config_path, "r") as f:
        path_config = yaml.safe_load(f)
    datasets = path_config["datasets"].keys()

    for target in iter_evaluation_targets(path_config, datasets):
        print(
            f"Evaluating dataset: {target.dataset}, model: {target.model}, run: {target.run}"
        )
        print(
            f"Reference shape: {target.ref.shape}, Generated shape: {target.generated.shape}"
        )
