import os

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def load_config(experiment: str | None = None, path: str | None = None) -> dict:
    if path is not None:
        config_path = path
    elif experiment is not None:
        config_path = os.path.join(CONFIG_DIR, f"{experiment}.yaml")
    else:
        raise ValueError("Provide experiment or path for config loading.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
