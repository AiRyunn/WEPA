# WEPA: Watermarks for Language Models via Probabilistic Automata

This repository contains the implementation of the paper [Watermarks for Language Models via Probabilistic Automata](https://arxiv.org/abs/2512.10185).


## Installation

Run the following commands to create a conda environment and install the required packages: 

```bash
conda create -n wepa python=3.12 -y && conda activate wepa
pip install -r requirements.txt
```

## Running Experiments

### Option 1: Jupyter Notebooks

Navigate to the `experiments/` directory and run the notebooks to reproduce results from the paper.

### Option 2: Python Scripts

Alternatively, execute the scripts in the `scripts/` directory directly from the command line.

```
python scripts/<script_name>.py
```

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
