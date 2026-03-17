#!/usr/bin/env bash
set -euo pipefail

python -m experiments.run --config experiments/configs/varying_length_long.yaml
