import time

import torch
from tqdm import tqdm

from experiments.utils import (
    build_runtime,
    get_wat_name,
    load_default_exp_watermarker,
    load_default_kgw_watermarker,
    load_default_wepa_watermarker,
)

experiment = "efficiency"

def run(config):
    print(config)
    max_length = config.get("max_length")
    lam = config.get("lam")
    n_samples = config.get("n_samples")
    seed = config.get("seed")
    device, model, tokenizer, data = build_runtime(config)
    dataset_size = len(data)

    vocab_size = tokenizer.vocab_size

    kgw_wat = load_default_kgw_watermarker(vocab_size, device, seed=seed)
    exp_wat = load_default_exp_watermarker(vocab_size, device, lam=lam, seed=seed)
    wepa_d1_wat = load_default_wepa_watermarker(
        vocab_size, device, lam=lam, degree=1, seed=seed
    )
    wepa_d2_wat = load_default_wepa_watermarker(
        vocab_size, device, lam=lam, degree=2, seed=seed
    )
    wepa_d2_b6_wat = load_default_wepa_watermarker(
        vocab_size, device, lam=lam, degree=2, bits=6, seed=seed
    )

    def evaluate_time(watermarker, unoptimized=False):
        tokens = torch.tensor(list(range(max_length)), device=device)
        if not unoptimized:
            # for JIT compilation
            for i in range(3):
                watermarker.p_value(tokens, n_samples=n_samples)

            tic = time.time()
            for i in tqdm(range(dataset_size)):
                watermarker.p_value(data[i]["input_ids"][0], n_samples=n_samples)
            toc = time.time()
        else:
            # for JIT compilation
            for i in range(3):
                watermarker.p_value_unoptimized(tokens, n_samples=n_samples)

            tic = time.time()
            for i in tqdm(range(dataset_size)):
                watermarker.p_value_unoptimized(data[i]["input_ids"][0], n_samples=n_samples)
            toc = time.time()

        elapsed = toc - tic
        label = f"{get_wat_name(watermarker)}{' (Unoptimized)' if unoptimized else ''}"
        print(f"Watermarker: {label}, Time taken: {elapsed:.3} s")
        return label, elapsed

    results = []
    results.append(evaluate_time(wepa_d1_wat))
    results.append(evaluate_time(wepa_d2_wat))
    results.append(evaluate_time(wepa_d2_b6_wat))
    results.append(evaluate_time(exp_wat))
    results.append(evaluate_time(exp_wat, unoptimized=True))
    results.append(evaluate_time(kgw_wat))

    print("\nResults:")
    for label, elapsed in results:
        print(f"- {label}: {elapsed:.3} s")

    metrics = {label: elapsed for label, elapsed in results}
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print(f"Completed {experiment}")
