from pathlib import Path

import torch
from tqdm import tqdm
import wandb

from experiments.utils import (
    build_runtime,
    estimate_conditional_entropy,
    generate_valid_sample,
    get_wat_name,
    load_default_dip_watermarker,
    load_default_exp_watermarker,
    load_default_kgw_watermarker,
    load_default_unbiased_watermarker,
    load_default_wepa_watermarker,
    load_unwatermarked_watermarker,
    log_wandb_run,
    save_text_results,
)

experiment = "diversity"


def _build_watermarkers(model, device, seed):
    wats = [
        load_default_wepa_watermarker(
            model.config.vocab_size,
            device,
            degree=2,
            bits=6,
            seed=seed,
        ),
        load_default_exp_watermarker(model.config.vocab_size, device, seed=seed),
        load_default_dip_watermarker(model.config.vocab_size, device, seed=seed),
        load_default_kgw_watermarker(model.config.vocab_size, device, seed=seed),
        load_default_unbiased_watermarker(model.config.vocab_size, device, seed=seed),
        load_unwatermarked_watermarker(),
    ]
    return {get_wat_name(wat): wat for wat in wats}


def _resolve_watermarker_names(watermarkers, configured_names=None):
    available_names = list(watermarkers.keys())
    if configured_names is None:
        return available_names

    unknown_names = sorted(set(configured_names) - set(available_names))
    if unknown_names:
        raise ValueError(
            "Unknown diversity watermarker(s): "
            f"{unknown_names}. Available options: {available_names}."
        )

    selected_names = []
    for name in configured_names:
        if name not in selected_names:
            selected_names.append(name)
    return selected_names


def _generate_prompt_samples(model, tokenizer, prompt, device, wat, max_length, samples_per_prompt):
    token_sequences = []
    texts = []

    for _ in range(samples_per_prompt):
        sample = generate_valid_sample(
            model=model,
            tokenizer=tokenizer,
            inputs=prompt,
            max_length=max_length,
            device=device,
            wat=wat,
        )
        if sample is None:
            continue
        token_sequences.append(sample.continuation_token_ids.tolist())
        texts.append(sample.continuation_text)

    return token_sequences, texts


def _save_sample_metrics(output_dir, model_name, wat_name, order, rows):
    output_path = (
        Path(output_dir)
        / f"{model_name.replace('/', '_')}_{wat_name}_k{order}_sample_metrics.csv"
    )
    save_text_results(rows, output_path)


def _select_longest_prompts(data, selected_prompt_count, candidate_pool_size):
    if selected_prompt_count < 1:
        raise ValueError(f"selected_prompt_count must be >= 1, got {selected_prompt_count}.")
    if candidate_pool_size < selected_prompt_count:
        raise ValueError(
            "candidate_pool_size must be >= selected_prompt_count, "
            f"got {candidate_pool_size} and {selected_prompt_count}."
        )
    if len(data) < candidate_pool_size:
        raise ValueError(
            f"Loaded {len(data)} prompts, but candidate_pool_size={candidate_pool_size}."
        )

    prompt_lengths = [
        (index, int(prompt["input_ids"].size(1)))
        for index, prompt in enumerate(data[:candidate_pool_size])
    ]
    prompt_lengths.sort(key=lambda item: (-item[1], item[0]))
    selected_indices = [index for index, _ in prompt_lengths[:selected_prompt_count]]
    return [data[index] for index in selected_indices], selected_indices


def run(config):
    print(config)

    seed = config.get("seed", 42)
    max_length = config.get("max_length")
    if max_length is None:
        raise ValueError("Config must include 'max_length'.")

    samples_per_prompt = config.get("samples_per_prompt")
    if samples_per_prompt is None or int(samples_per_prompt) < 1:
        raise ValueError("Config must include 'samples_per_prompt' >= 1.")
    samples_per_prompt = int(samples_per_prompt)

    k_values = config.get("k_values")
    if not k_values:
        raise ValueError("Config must include a non-empty 'k_values' list.")

    output_dir = config.get("output_dir", "artifacts/output/diversity")
    configured_watermarker_names = config.get("watermarkers")
    selected_prompt_count = int(config.get("dataset_size", 1))
    candidate_pool_size = int(config.get("prompt_candidate_pool_size", selected_prompt_count))

    runtime_config = dict(config)
    runtime_config["dataset_size"] = candidate_pool_size
    device, model, tokenizer, data = build_runtime(runtime_config)
    data, selected_prompt_indices = _select_longest_prompts(
        data,
        selected_prompt_count=selected_prompt_count,
        candidate_pool_size=candidate_pool_size,
    )
    dataset_size = len(data)

    print(
        f"Selected {dataset_size} prompt(s) from the first {candidate_pool_size} samples: "
        f"indices={selected_prompt_indices}"
    )

    watermarkers = _build_watermarkers(model, device, seed)
    selected_watermarker_names = _resolve_watermarker_names(
        watermarkers,
        configured_watermarker_names,
    )

    for wat_name in selected_watermarker_names:
        wat = watermarkers[wat_name]
        print(f"Running diversity experiment for {wat_name}")

        per_prompt_samples = []
        for prompt_index in tqdm(range(dataset_size), desc=f"{wat_name} prompts"):
            token_sequences, texts = _generate_prompt_samples(
                model=model,
                tokenizer=tokenizer,
                prompt=data[prompt_index],
                device=device,
                wat=wat,
                max_length=max_length,
                samples_per_prompt=samples_per_prompt,
            )
            if not token_sequences:
                print(f"Prompt {prompt_index}: no valid samples generated; skipping.")
                continue

            per_prompt_samples.append(
                {
                    "prompt_index": prompt_index,
                    "token_sequences": token_sequences,
                    "texts": texts,
                }
            )

        if not per_prompt_samples:
            print(f"No valid samples generated for {wat_name}; skipping run.")
            continue

        for order in k_values:
            order = int(order)
            sample_rows = []
            entropy_values = []
            transition_counts = []

            for prompt_result in per_prompt_samples:
                for sample_index, token_sequence in enumerate(prompt_result["token_sequences"]):
                    try:
                        estimate = estimate_conditional_entropy([token_sequence], order)
                    except ValueError:
                        continue

                    row = {
                        "prompt_index": prompt_result["prompt_index"],
                        "sample_index": sample_index,
                        "order": order,
                        "conditional_entropy_bits": estimate["conditional_entropy_bits"],
                        "total_transitions": estimate["total_transitions"],
                        "n_contexts": estimate["n_contexts"],
                    }
                    sample_rows.append(row)
                    entropy_values.append(estimate["conditional_entropy_bits"])
                    transition_counts.append(estimate["total_transitions"])

            if not sample_rows:
                print(f"No valid sample-level entropy estimates for {wat_name} at k={order}.")
                continue

            entropy_tensor = torch.tensor(entropy_values, dtype=torch.float32)
            transition_tensor = torch.tensor(transition_counts, dtype=torch.float32)

            metrics = {
                "conditional_entropy_bits_mean": float(torch.mean(entropy_tensor)),
                "conditional_entropy_bits_median": float(torch.median(entropy_tensor)),
                "conditional_entropy_bits_min": float(torch.min(entropy_tensor)),
                "conditional_entropy_bits_max": float(torch.max(entropy_tensor)),
                "transitions_mean": float(torch.mean(transition_tensor)),
                "transitions_median": float(torch.median(transition_tensor)),
                "n_samples": len(sample_rows),
                "n_prompts": len(per_prompt_samples),
                "samples_per_prompt": samples_per_prompt,
            }
            tables = {
                "sample_entropy": wandb.Table(
                    data=[
                        [
                            row["prompt_index"],
                            row["sample_index"],
                            row["order"],
                            row["conditional_entropy_bits"],
                            row["total_transitions"],
                            row["n_contexts"],
                        ]
                        for row in sample_rows
                    ],
                    columns=[
                        "prompt_index",
                        "sample_index",
                        "order",
                        "conditional_entropy_bits",
                        "total_transitions",
                        "n_contexts",
                    ],
                )
            }

            wandb_config = {
                "experiment": experiment,
                "dataset_size": dataset_size,
                "model": model.name_or_path,
                "algo": wat_name,
                "order": order,
                "max_length": max_length,
                "samples_per_prompt": samples_per_prompt,
                "device": str(device),
            }
            log_wandb_run(
                experiment=experiment,
                run_name=f"{model.name_or_path}_{wat_name}_k{order}",
                wandb_config=wandb_config,
                metrics=metrics,
                tables=tables,
            )
            _save_sample_metrics(
                output_dir=output_dir,
                model_name=model.name_or_path,
                wat_name=wat_name,
                order=order,
                rows=sample_rows,
            )

            print(f"k={order}")
            for key, value in metrics.items():
                print(f"  {key}: {value}")

    print(f"Completed {experiment}")
