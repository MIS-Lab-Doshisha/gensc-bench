import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.graph_util import get_topn_adj_list
from common.path_manager import PathManager
from common.util import (
    extract_tensor_from_dataloaders,
    get_reconstruction_from_npz,
    setup_logger,
)
from dataset.graph_dataset import GraphDataset, create_dataloaders, get_dataloaders
from eval.eval_func import (
    evaluate_graphs,
    get_best_thresholded_matrices,
    get_planarity_rate,
)
from eval.graph_mmd import strength_stats
from eval.plot import plot_mean_matrices


def thresh():
    manager = PathManager("path_list.json")
    keys = manager.get_all_keys()
    logger = setup_logger("thresh", "log/thresh.log")

    for key in keys:
        target_path = manager.get_original_path(key)
        target = get_reconstruction_from_npz(target_path)
        recons = (Path("checkpoints/artificial") / key).rglob("generated_graphs.npz")
        for recon_path in recons:
            recon = get_reconstruction_from_npz(recon_path)
            logger.info(f"Processing {target_path} and {recon_path}")

            thresh = np.linspace(0.1, 0.49, 40)
            thresh = np.round(thresh, decimals=2)
            matrices = get_best_thresholded_matrices(target, recon, thresh_list=thresh)

            save_path = Path("output/thresholded") / key / recon_path.parent.name
            save_path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_path / "generated_graphs.npz", matrices)


def eval_mmds():
    logger = setup_logger("eval", "log/evaluation.log", mode="a")
    manager = PathManager("processed_path_list.json")

    keys = manager.get_all_keys()

    for key in keys:
        original = manager.get_original_path(key)
        csv_path = f"output/mmd_results/{key}_mmds.csv"
        dataset = GraphDataset(original)
        train, val, test = get_dataloaders(dataset)
        ref = extract_tensor_from_dataloaders([test]).numpy()

        results = []
        pathlists = manager.get_all_recon_paths(key)

        for path in pathlists:
            logger.info(f"Evaluating graphs from: {original}, {path}")
            generated = get_reconstruction_from_npz(path)
            mmds = evaluate_graphs(ref, generated, logger)
            row_data = {"Model": Path(path).parent.name.upper()}

            row_data.update(mmds)

            results.append(row_data)

        df = pd.DataFrame(results)
        columns = [
            "Model",
            "Top n Edges",
            "Degree MMD",
            "Clustering MMD",
            "Orbit MMD",
            "Spectral MMD",
        ]
        df = df.reindex(columns=columns)

        df.to_csv(csv_path, mode="w")
        print(f"Saved evaluation results to {csv_path}")


def eval_strength_mmd():
    manager = PathManager("processed_path_list.json")
    keys = ["sc"]

    for key in keys:
        csv_path = f"output/mmd_results/{key}_mmds.csv"
        df = pd.read_csv(csv_path)
        all_data = []
        train, val, test = create_dataloaders(key)
        test_data = extract_tensor_from_dataloaders([test]).numpy()

        models = manager.get_all_models(key)
        for model in models:
            recon_path = manager.get_recon_path(key, model)
            recon_data = get_reconstruction_from_npz(recon_path)

            if key == "sc":
                topns = [0.05, 0.1, 0.2, 0.3]
                for top_n in topns:
                    test_data_thresh = get_topn_adj_list(test_data, top_n=top_n)
                    recon_data = get_topn_adj_list(recon_data, top_n=top_n)
                    mmds = strength_stats(test_data_thresh, recon_data)
                    all_data.append(
                        {"Model": model.upper(), "Top N": top_n, "Strength MMD": mmds}
                    )
                    print(
                        f"Dataset: {key}, Model: {model}, Top N: {top_n}, Strength MMD: {mmds}"
                    )
            else:
                mmds = strength_stats(test_data, recon_data)
                all_data.append({"Model": model.upper(), "Strength MMD": mmds})
                print(f"Dataset: {key}, Model: {model}, Strength MMD: {mmds}")

        strength_df = pd.DataFrame(all_data)
        if key == "sc":
            merge_keys = ["Model", "Top N"]
        else:
            merge_keys = ["Model"]

        try:
            df = df.merge(strength_df, how="left", on=merge_keys)
        except Exception as e:
            print(f"Error merging strength MMDs for dataset {key}: {e}")
            continue

        df.to_csv(f"{csv_path}_w", mode="w", index=False)
        print(f"Saved updated MMD results with Strength MMD to {csv_path}_w")


def eval_mean():
    manager = PathManager("processed_path_list.json")
    keys = manager.get_all_keys()

    for key in keys:
        train, val, test = create_dataloaders(key)
        test_data = extract_tensor_from_dataloaders([test]).numpy()

        data_dict = {}
        models = manager.get_all_models(key)
        for model in models:
            recon_path = manager.get_recon_path(key, model)
            recon_data = get_reconstruction_from_npz(recon_path)

            data_dict[model.upper()] = recon_data

        plot_mean_matrices(
            test_data, data_dict, output_path=f"output/mean_matrices/{key}_mean.pdf"
        )


def eval_planarity():
    manager = PathManager("processed_path_list.json")
    key = "planar"
    models = manager.get_all_models(key)

    all_data = []

    for model in models:
        recon_path = manager.get_recon_path(key, model)
        recon_data = get_reconstruction_from_npz(recon_path)

        planarity = get_planarity_rate(recon_data)
        print(f"Dataset: {key}, Model: {model}, Planarity Rate: {planarity:.4f}")

        all_data.append({"Model": model.upper(), "Planarity Rate": planarity})

    df = pd.DataFrame(all_data)
    csv_path = "output/mmd_results/planarity_rates.csv"
    df.to_csv(csv_path, mode="w", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mmds", action="store_true", help="Evaluate MMDs for generated graphs"
    )
    parser.add_argument(
        "--thresh", action="store_true", help="Threshold generated graphs"
    )
    parser.add_argument(
        "--mean", action="store_true", help="Plot mean adjacency matrices"
    )
    parser.add_argument(
        "--strength", action="store_true", help="Evaluate Strength MMDs"
    )
    parser.add_argument(
        "--planarity", action="store_true", help="Evaluate Planarity Rates"
    )

    args = parser.parse_args()
    if args.mmds:
        eval_mmds()
    if args.thresh:
        thresh()
    if args.mean:
        eval_mean()
    if args.strength:
        eval_strength_mmd()
    if args.planarity:
        eval_planarity()
