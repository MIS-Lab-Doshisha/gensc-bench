from pathlib import Path

import networkx as nx
import pandas as pd
import powerlaw

from common.graph_util import get_topn_adj_list, get_topns
from common.path_manager import PathManager
from common.util import get_reconstruction_from_npz
from dataset.graph_dataset import get_testdata_as_numpy
from eval.eval_func import calc_modularities, get_edge_jaccard, get_hubs_spearman
from eval.graph_mmd import degree_worker
from eval.plot import plot_common_edges


def compute_modularities():
    output_dir = Path("output/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = ["sbm"]
    models = ["vae", "gan", "ddpm"]
    manager = PathManager("processed_path_list.json")
    df_rows = []
    # TODO: まずはcsvを保存する。その後plot.pyで読み込む形にする。

    for key in keys:
        data = {}
        print(f"Dataset: {key}")
        test = get_testdata_as_numpy(key)

        data["real"] = test

        for model in models:
            recon_path = manager.get_recon_path(key, model)
            reconstructions = get_reconstruction_from_npz(recon_path)

            data[model] = reconstructions

        top_ns = get_topns(key)

        for top_n in top_ns:
            for model in models + ["real"]:
                matrices = data[model]
                matrices_thresholded = get_topn_adj_list(matrices, top_n=top_n)
                modularities = calc_modularities(matrices_thresholded)

                for score in modularities:
                    df_rows.append(
                        {
                            "dataset": key,
                            "model": model,
                            "modularity": score,
                            "top_n": top_n,
                        }
                    )

    df_all = pd.DataFrame(df_rows)
    df_all.to_csv(output_dir / "modularity_scores.csv", index=False, mode="a")


def get_and_plot_jaccard():
    output_dir = Path("output/metrics/jaccard")
    output_dir.mkdir(parents=True, exist_ok=True)

    test = get_testdata_as_numpy("sc")
    models = ["vae", "gan", "ddpm"]
    manager = PathManager("processed_path_list.json")

    ns = [0.01, 0.05, 0.1]
    all_data = []

    test_mean = test.mean(axis=0)

    for n in ns:
        model_results_for_plot = []

        for model in models:
            path = manager.get_recon_path("sc", model)
            recon = get_reconstruction_from_npz(path)

            recon_mean = recon.mean(axis=0)

            jaccard = get_edge_jaccard(test_mean, recon_mean, n=n)

            all_data.append(
                {
                    "model": model,
                    "n": n,
                    "jaccard": jaccard,
                }
            )

            model_results_for_plot.append(
                {"name": model.upper(), "adj": recon_mean, "jaccard": jaccard}
            )

        plot_filename = f"comparison_n_{str(n).replace('.', '')}.pdf"

        print(f"Plotting for n={n}...")
        plot_common_edges(
            adj_target=test_mean,
            model_results=model_results_for_plot,
            n=n,
            output_path=output_dir / plot_filename,
        )

    df = pd.DataFrame(all_data)
    df.to_csv(output_dir / "jaccard_scores.csv", index=False)


def get_strength_spearman():
    output_dir = Path("output/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    test = get_testdata_as_numpy("sc")
    models = ["vae", "gan", "ddpm"]
    manager = PathManager("processed_path_list.json")

    all_data = []

    for model in models:
        path = manager.get_recon_path("sc", model)
        recon = get_reconstruction_from_npz(path)

        test_mean = test.mean(axis=0)
        recon_mean = recon.mean(axis=0)

        corr, p = get_hubs_spearman(test_mean, recon_mean)

        all_data.append(
            {
                "model": model,
                "spearman_corr": corr,
                "p_value": p,
            }
        )

    df = pd.DataFrame(all_data)
    df.to_csv(output_dir / "hub_spearman.csv", index=False)


def get_powerlaw():
    keys = ["ba", "wba", "sc"]
    manager = PathManager("processed_path_list.json")

    for key in keys:
        models = manager.get_all_models(key, without_ref=True) + ["real"]
        all_data = []

        for model in models:
            if model == "real":
                graph_data = get_testdata_as_numpy(key)
            else:
                recon_path = manager.get_recon_path(key, model)
                graph_data = get_reconstruction_from_npz(recon_path)

            top_ns = get_topns(key)
            for top_n in top_ns:
                graph_data_thresh = get_topn_adj_list(graph_data, top_n=top_n)

                for adj in graph_data_thresh:
                    degree = degree_worker(nx.Graph(adj))
                    fit = powerlaw.Fit(degree, verbose=False, discrete=True)
                    r, p = fit.distribution_compare("power_law", "lognormal")

                    all_data.append(
                        {
                            "Model": model,
                            "top n": top_n,
                            "alpha": fit.power_law.alpha,
                            "xmin": fit.power_law.xmin,
                            "R": r,
                            "P": p,
                        }
                    )

        df_all = pd.DataFrame(all_data)
        df_all.to_csv(f"output/metrics/powerlaw/{key}_powerlaw.csv", index=False)


if __name__ == "__main__":
    get_powerlaw()
    get_and_plot_jaccard()
    get_strength_spearman()
    compute_modularities()
