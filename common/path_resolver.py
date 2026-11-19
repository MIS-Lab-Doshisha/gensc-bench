from pathlib import Path


def get_generated_path(config, dataset, model, run):
    return (
        Path(config["roots"]["checkpoints"])
        / dataset
        / model
        / run
        / "generated_graphs.npz"
    )


def get_thresholded_path(config, dataset, model, run):
    return (
        Path(config["roots"]["thresholded"])
        / dataset
        / model
        / run
        / "thresholded_matrices.npz"
    )


def get_graph_path(config, dataset, model, run):
    if config["datasets"][dataset]["thresholding"]:
        return get_thresholded_path(config, dataset, model, run)
    else:
        return get_generated_path(config, dataset, model, run)
