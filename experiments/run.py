import argparse
import importlib
import os

from experiments.utils import load_config


def run_all_experiments(config: dict) -> None:
    output_dir = os.path.join("artifacts", "output")
    os.makedirs(output_dir, exist_ok=True)

    experiment = config.get("name")
    if not experiment:
        raise ValueError("Config must include a 'name' field.")
    parameters_list = config.get("parameters", [])

    module_name = f"experiments.runs.{experiment}"
    module = importlib.import_module(module_name)
    run_func = getattr(module, "run", None)
    if run_func is None or not callable(run_func):
        raise ValueError(f"Module {module_name} has no run()")

    common = config.get("common", {})
    for params in parameters_list:
        run_func(common | params)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run experiment runners.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args(argv)

    config = load_config(path=args.config)
    run_all_experiments(config)


if __name__ == "__main__":
    main()
