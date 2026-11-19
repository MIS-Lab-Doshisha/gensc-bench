from pathlib import Path

import numpy as np


def make_npz_from_csv(csv_path: str, npz_path_stem: str, delimiter: str = ","):
    csv_dir = Path(csv_path)

    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    csv_files = sorted(csv_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {csv_path}")

    arrays = []
    expected_shape = None

    for file in csv_files:
        array = np.loadtxt(file, delimiter=delimiter)
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: {file} has shape {array.shape}, expected {expected_shape}"
            )
        arrays.append(array)

    data = np.stack(arrays, axis=0)

    Path(npz_path_stem).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{npz_path_stem}/hcp_raw_data.npz", adj_matrix=data)

    print(f"Saved: {npz_path_stem}/hcp_raw_data.npz with shape {data.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the directory containing CSV files.",
    )
    parser.add_argument(
        "--npz_path_stem",
        type=str,
        required=True,
        help="Path stem for the output NPZ file.",
    )
    parser.add_argument(
        "--delimiter", type=str, default=",", help="Delimiter used in the CSV files."
    )

    args = parser.parse_args()

    make_npz_from_csv(args.csv_path, args.npz_path_stem, args.delimiter)
