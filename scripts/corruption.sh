#!/usr/bin/env bash
set -euo pipefail

python -m experiments.run --config experiments/configs/corruption.yaml
