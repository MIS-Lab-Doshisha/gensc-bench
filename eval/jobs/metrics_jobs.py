from pathlib import Path

import numpy as np
import pandas as pd

from common.graph_util import get_topn_adj_list, get_topns
from common.visualize import get_formatted_model_name
from eval.core.graph_statistics import calc_modularities
from eval.core.iteration import iter_evaluation_targets
from eval.core.similaritiy import get_edge_jaccard, get_hubs_spearman
from eval.jobs.plot import plot_common_edges


def compute_modularities(path_config, datasets):
    output_dir = Path(f"{path_config['roots']['output']}/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_rows = []
    test_data = {}

    for target in iter_evaluation_targets(path_config, datasets, include_real=False):
        print(f"Dataset: {target.dataset}")
        test_data.setdefault(target.dataset, target.ref)

        for top_n in get_topns(target.dataset):
            matrices_thresholded = get_topn_adj_list(target.generated, top_n=top_n)
            modularities = calc_modularities(matrices_thresholded)

            for score in modularities:
                df_rows.append(
                    {
                        "dataset": target.dataset,
                        "model": target.model,
                        "run": target.run,
                        "top_n": top_n,
                        "modularity": score,
                    }
                )

    for dataset, matrices in test_data.items():
        for top_n in get_topns(dataset):
            matrices_thresholded = get_topn_adj_list(matrices, top_n=top_n)
            modularities = calc_modularities(matrices_thresholded)

            for score in modularities:
                df_rows.append(
                    {
                        "dataset": dataset,
                        "model": "real",
                        "run": None,
                        "top_n": top_n,
                        "modularity": score,
                    }
                )

    df_all = pd.DataFrame(df_rows)
    df_all.to_csv(output_dir / "modularity_scores.csv", index=False, mode="w")


def get_and_plot_jaccard(path_config: dict, datasets):
    output_dir_root = Path(f"{path_config['roots']['output']}/metrics/jaccard")
    ns = [0.01, 0.05, 0.1]
    all_data = []

    for dataset in datasets:
        output_dir = output_dir_root / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_by_model = {model: {} for model in path_config["models"]}

        for target in iter_evaluation_targets(
            path_config, [dataset], include_real=False
        ):
            print(
                f"Evaluating dataset: {target.dataset}, model: {target.model}, run: {target.run}"
            )

            test_mean = target.ref.mean(axis=0)
            generated_mean = target.generated.mean(axis=0)
            generated_by_model[target.model][target.run] = generated_mean

        for n in ns:
            mean_results = []

            for model, run in generated_by_model.items():
                run_means = list(run.values())
                model_mean = np.mean(np.stack(run_means, axis=0), axis=0)

                jaccard = get_edge_jaccard(test_mean, model_mean, n=n)
                mean_results.append(
                    {
                        "name": get_formatted_model_name(model),
                        "adj": model_mean,
                        "jaccard": jaccard,
                    }
                )

                all_data.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "run": "mean",
                        "n": n,
                        "jaccard": jaccard,
                    }
                )

            plot_common_edges(
                adj_target=test_mean,
                model_results=mean_results,
                n=n,
                output_path=output_dir / f"comparison_n_{str(n).replace('.', '')}.pdf",
            )

            run_rows = []
            row_labels = []

            for model, run_data in generated_by_model.items():
                run_results = []

                for run, generated_mean in run_data.items():
                    jaccard = get_edge_jaccard(test_mean, generated_mean, n=n)

                    run_index = run if isinstance(run, int) else int(run.split("_")[-1])
                    run_results.append(
                        {
                            "name": f"run_{run_index + 1}",
                            "adj": generated_mean,
                            "jaccard": jaccard,
                        }
                    )

                    all_data.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "run": run,
                            "n": n,
                            "jaccard": jaccard,
                        }
                    )

                run_rows.append(run_results)
                row_labels.append(get_formatted_model_name(model))

            plot_common_edges(
                adj_target=test_mean,
                model_results=run_rows,
                n=n,
                output_path=output_dir
                / f"comparison_runs_n_{str(n).replace('.', '')}.pdf",
                row_labels=row_labels,
            )

    df = pd.DataFrame(all_data)
    df.to_csv(output_dir_root / "jaccard_scores.csv", index=False)


def get_strength_spearman(path_config, datasets):
    output_dir = Path(f"{path_config['roots']['output']}/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    for target in iter_evaluation_targets(path_config, datasets, include_real=False):
        print(
            f"Evaluating dataset: {target.dataset}, model: {target.model}, run: {target.run}"
        )

        test_mean = target.ref.mean(axis=0)
        generated_mean = target.generated.mean(axis=0)

        corr, p = get_hubs_spearman(test_mean, generated_mean)

        all_data.append(
            {
                "dataset": target.dataset,
                "model": target.model,
                "run": target.run,
                "spearman_corr": corr,
                "p_value": p,
            }
        )

    df = pd.DataFrame(all_data)
    df.to_csv(output_dir / "hub_spearman.csv", index=False)
    print(f"Saved evaluation results to {output_dir / 'hub_spearman.csv'}")
