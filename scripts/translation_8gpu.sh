#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-experiments/configs/translation.yaml}"
NUM_SHARDS=8

for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES="${shard_index}" \
    WEPA_TRANSLATION_SHARD_INDEX="${shard_index}" \
    WEPA_TRANSLATION_NUM_SHARDS="${NUM_SHARDS}" \
    python -m experiments.run --config "${CONFIG}" &
done

wait
