# WEPA: Watermarks for Language Models via Probabilistic Automata

This repository contains the implementation of the paper [Watermarks for Language Models via Probabilistic Automata](https://arxiv.org/abs/2512.10185).


## Installation

Run the following commands to create a conda environment and install the required packages: 

```bash
conda create -n wepa python=3.12 -y && conda activate wepa
pip install -e .
```

## Running Experiments

### Option 1: Jupyter Notebooks

Notebooks live in `notebooks/` and are intended for manual exploration. Each notebook loads its own config from `experiments/configs/<experiment>.yaml`, prints it, and calls the corresponding runner in `experiments/runs/`.

### Option 2: Bash Scripts

Execute the bash scripts in the `scripts/` directory directly from the command line.

```
./scripts/<experiment>.sh
```

For translation, you can also launch 8 shard workers in parallel, one per GPU:

```bash
./scripts/translation_8gpu.sh
```

The CLI now accepts config paths only:

```bash
python -m experiments.run --config experiments/configs/ppl.yaml
```

Configs live in `experiments/configs/`.

## Layout

- `src/watermarker/`: canonical package code
- `experiments/runs/`: runnable experiment runners
- `experiments/configs/`: per-experiment YAML configs
- `notebooks/`: manual notebooks (thin wrappers around scripts)
- `scripts/`: bash scripts for common runs
- `tests/`: automated validation
- `artifacts/`: generated outputs, caches, and WandB runs

## Citation

If you find this work useful in your research, please consider citing the paper:

```bibtex
@article{wang2025watermarks,
  title={Watermarks for Language Models via Probabilistic Automata},
  author={Wang, Yangkun and Shang, Jingbo},
  journal={arXiv preprint arXiv:2512.10185},
  year={2025}
}
```
