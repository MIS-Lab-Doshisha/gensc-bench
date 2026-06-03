import json
from pathlib import Path
from typing import List


class PathManager:
    def __init__(self, json_path: str | Path):
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Error: {json_path} not found.")

        with json_path.open("r") as f:
            self.json = json.load(f)

    def get_original_path(self, key: str) -> str:
        try:
            path = self.json[key]["original"]
            return path
        except KeyError:
            raise KeyError(f"Key '{key}' not found in JSON.")

    def get_recon_path(self, key: str, model: str) -> str:
        try:
            recon = self.json[key]["reconstructions"][model]
            return recon
        except KeyError:
            if key not in self.json:
                raise KeyError(f"Key '{key}' not found in JSON.")
            if model not in self.json[key]["reconstructions"]:
                raise KeyError(f"Model '{model}' not found for key '{key}' in JSON.")
            raise

    def get_all_recon_paths(self, key: str) -> List[str]:
        try:
            recons = self.json[key]["reconstructions"]
            return list(recons.values())
        except KeyError:
            raise KeyError(f"Key '{key}' not found in JSON.")

    def get_all_original_paths(self) -> List[str]:
        originals = []
        for key in self.json:
            originals.append(self.json[key]["original"])
        return originals

    def get_all_keys(self) -> List[str]:
        return list(self.json.keys())

    def get_all_models(self, key: str, without_ref: bool = False) -> List[str]:
        try:
            recons = self.json[key]["reconstructions"]
            if without_ref:
                return [model for model in recons.keys() if model != "ref"]
            return list(recons.keys())
        except KeyError:
            raise KeyError(f"Key '{key}' not found in JSON.")
