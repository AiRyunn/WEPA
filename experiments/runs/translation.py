import os
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
import wandb

from experiments.utils import (
    UnbiasedWatermarker,
    build_runtime,
    calc_roc_auc_score,
    calc_tpr_at_fpr,
    generate_valid_sample,
    get_wat_name,
    log_wandb_run,
    load_default_dip_watermarker,
    load_seq2seq_model_and_tokenizer,
    load_default_exp_watermarker,
    load_default_kgw_watermarker,
    load_default_unbiased_watermarker,
    load_default_wepa_watermarker,
    save_text_results,
    translate_texts,
)

experiment = "translation"

ROUND_TRIP_MODELS = {
    "fr": (
        "Helsinki-NLP/opus-mt-tc-big-en-fr",
        "Helsinki-NLP/opus-mt-tc-big-fr-en",
    ),
    "ru": (
        "Helsinki-NLP/opus-mt-en-ru",
        "Helsinki-NLP/opus-mt-ru-en",
    ),
}


@dataclass
class TranslationPipeline:
    forward_model: torch.nn.Module
    forward_tokenizer: object
    backward_model: torch.nn.Module
    backward_tokenizer: object


@dataclass(frozen=True)
class TranslationJob:
    language: str
    wat_name: str
    max_length: int


def _load_translation_pipeline(language, device):
    forward_name, backward_name = ROUND_TRIP_MODELS[language]
    forward_model, forward_tokenizer = load_seq2seq_model_and_tokenizer(forward_name, device)
    backward_model, backward_tokenizer = load_seq2seq_model_and_tokenizer(backward_name, device)
    return TranslationPipeline(
        forward_model=forward_model,
        forward_tokenizer=forward_tokenizer,
        backward_model=backward_model,
        backward_tokenizer=backward_tokenizer,
    )


def _generate_samples(model, tokenizer, data, device, max_length, wat=None):
    generated_texts = []
    generated_token_ids = []
    prompt_token_ids = []

    for i in tqdm(range(len(data))):
        sample = generate_valid_sample(
            model=model,
            tokenizer=tokenizer,
            inputs=data[i],
            max_length=max_length,
            device=device,
            wat=wat,
        )
        if sample is None:
            continue

        generated_texts.append(sample.continuation_text)
        generated_token_ids.append(sample.continuation_token_ids)
        prompt_token_ids.append(sample.prompt_token_ids)

    return generated_texts, generated_token_ids, prompt_token_ids


def _round_trip_texts(texts, pipeline, device, batch_size, max_source_length, max_target_length):
    pivot_texts = translate_texts(
        texts,
        pipeline.forward_model,
        pipeline.forward_tokenizer,
        device,
        batch_size=batch_size,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    round_trip_texts = translate_texts(
        pivot_texts,
        pipeline.backward_model,
        pipeline.backward_tokenizer,
        device,
        batch_size=batch_size,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    return pivot_texts, round_trip_texts


def _retokenize_texts(tokenizer, texts, max_length):
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    return [torch.tensor(token_ids, dtype=torch.long) for token_ids in encoded]


def _evaluate_attacked_samples(
    wat,
    model,
    tokenizer,
    attacked_texts,
    prompt_token_ids,
    max_retokenized_length,
    n_samples,
    upperbound,
):
    attacked_token_ids = _retokenize_texts(tokenizer, attacked_texts, max_retokenized_length)
    model_device = next(model.parameters()).device

    p_values = []
    pos_scores = []
    attacked_lengths = []
    for prompt_ids, token_ids in zip(prompt_token_ids, attacked_token_ids):
        if len(token_ids) == 0:
            continue
        attacked_lengths.append(int(len(token_ids)))
        if isinstance(wat, UnbiasedWatermarker):
            prefix = prompt_ids[-wat.config.prefix_length :]
            eval_token_ids = torch.cat([prefix, token_ids], dim=0).unsqueeze(0).to(model_device)
            p_value = wat.p_value(eval_token_ids, model=model)
            pos_score = wat.test_statistic(model, eval_token_ids)
        else:
            eval_token_ids = token_ids.to(model_device)
            if upperbound:
                p_value = wat.p_value(eval_token_ids, n_samples=n_samples, upperbound=True)
            else:
                p_value = wat.p_value(eval_token_ids, n_samples=n_samples)
            pos_score = wat.test_statistic(eval_token_ids)
        p_values.append(float(p_value))
        pos_scores.append(float(pos_score))

    return p_values, pos_scores, attacked_lengths


def _save_examples(
    output_dir,
    model_name,
    wat_name,
    language,
    max_length,
    original_texts,
    pivot_texts,
    round_trip_texts,
    attacked_lengths,
    n_examples,
):
    rows = []
    for original_text, pivot_text, round_trip_text, attacked_length in zip(
        original_texts[:n_examples],
        pivot_texts[:n_examples],
        round_trip_texts[:n_examples],
        attacked_lengths[:n_examples],
    ):
        rows.append(
            {
                "original_text": original_text,
                "pivot_text": pivot_text,
                "round_trip_text": round_trip_text,
                "attacked_token_length": attacked_length,
            }
        )

    output_path = (
        Path(output_dir)
        / f"{model_name.replace('/', '_')}_{wat_name}_{language}_len{max_length}_examples.csv"
    )
    save_text_results(rows, output_path)


def _resolve_sharding(config):
    shard_index = int(os.environ.get("WEPA_TRANSLATION_SHARD_INDEX", config.get("shard_index", 0)))
    num_shards = int(os.environ.get("WEPA_TRANSLATION_NUM_SHARDS", config.get("num_shards", 1)))

    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}], got {shard_index}.")

    return shard_index, num_shards


def _build_jobs(languages, wat_names, max_lengths):
    return [
        TranslationJob(language=language, wat_name=wat_name, max_length=max_length)
        for language in languages
        for wat_name in wat_names
        for max_length in max_lengths
    ]


def _resolve_watermarker_names(watermarkers, configured_names=None):
    available_names = list(watermarkers.keys())
    if configured_names is None:
        return available_names

    unknown_names = sorted(set(configured_names) - set(available_names))
    if unknown_names:
        raise ValueError(
            "Unknown translation watermarker(s): "
            f"{unknown_names}. Available options: {available_names}."
        )

    selected_names = []
    for name in configured_names:
        if name not in selected_names:
            selected_names.append(name)
    return selected_names


def run(config):
    print(config)

    seed = config.get("seed", 42)
    model_name = config.get("model_name")
    batch_size = config.get("batch_size", 8)
    max_source_length = config.get("max_source_length", 512)
    max_target_length = config.get("max_target_length", 512)
    max_retokenized_length = config.get("max_retokenized_length", 512)
    evaluate_scores = config.get("evaluate_scores", True)
    output_dir = config.get("output_dir", "artifacts/output/translation")
    languages = config.get("languages", ["fr", "ru"])
    max_lengths = config.get("max_lengths", [25, 50, 75, 100, 125, 150, 175, 200, 225, 250])
    n_samples = config.get("n_samples", 10000)
    upperbound = config.get("upperbound", False)
    n_examples = config.get("n_examples", 3)
    configured_watermarker_names = config.get("watermarkers")
    shard_index, num_shards = _resolve_sharding(config)

    device, model, tokenizer, data = build_runtime(config)
    dataset_size = len(data)

    wats = [
        load_default_wepa_watermarker(model.config.vocab_size, device, degree=1, seed=seed),
        load_default_wepa_watermarker(model.config.vocab_size, device, degree=2, seed=seed),
        load_default_exp_watermarker(model.config.vocab_size, device, seed=seed),
        load_default_dip_watermarker(model.config.vocab_size, device, seed=seed),
        load_default_kgw_watermarker(model.config.vocab_size, device, seed=seed),
        load_default_unbiased_watermarker(model.config.vocab_size, device, seed=seed),
    ]
    watermarkers = {get_wat_name(wat): wat for wat in wats}
    selected_watermarker_names = _resolve_watermarker_names(
        watermarkers,
        configured_watermarker_names,
    )
    jobs = _build_jobs(languages, selected_watermarker_names, max_lengths)
    shard_jobs = jobs[shard_index::num_shards]

    print(
        f"Translation shard {shard_index + 1}/{num_shards} on {device}: "
        f"{len(shard_jobs)} of {len(jobs)} jobs."
    )

    pipelines = {}
    for job in shard_jobs:
        if job.language not in pipelines:
            print(f"Loading translation pipeline for {job.language}")
            pipelines[job.language] = _load_translation_pipeline(job.language, device)

    for job in shard_jobs:
        wat = watermarkers[job.wat_name]
        pipeline = pipelines[job.language]
        print(
            f"Running translation attack: shard={shard_index}/{num_shards}, "
            f"wat={job.wat_name}, language={job.language}, max_length={job.max_length}"
        )

        original_texts, _, prompt_token_ids = _generate_samples(
            model=model,
            tokenizer=tokenizer,
            data=data,
            device=device,
            max_length=job.max_length,
            wat=wat,
        )
        pivot_texts, round_trip_texts = _round_trip_texts(
            original_texts,
            pipeline,
            device,
            batch_size=batch_size,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )
        p_values, pos_scores, attacked_lengths = _evaluate_attacked_samples(
            wat=wat,
            model=model,
            tokenizer=tokenizer,
            attacked_texts=round_trip_texts,
            prompt_token_ids=prompt_token_ids,
            max_retokenized_length=max_retokenized_length,
            n_samples=n_samples,
            upperbound=upperbound,
        )
        if not p_values:
            print("No usable attacked samples after round-trip translation; skipping run.")
            continue

        metrics = {
            "p_value_median": float(torch.median(torch.tensor(p_values))),
            "p_value_mean": float(torch.mean(torch.tensor(p_values))),
            "p_value_min": float(torch.min(torch.tensor(p_values))),
            "p_value_max": float(torch.max(torch.tensor(p_values))),
            "power_at_0.05": float(sum(p < 0.05 for p in p_values) / len(p_values)),
            "attacked_length_mean": float(
                torch.mean(torch.tensor(attacked_lengths, dtype=torch.float32))
            ),
            "attacked_length_median": float(
                torch.median(torch.tensor(attacked_lengths, dtype=torch.float32))
            ),
            "n_samples": len(p_values),
        }
        tables = {
            "p_values": wandb.Table(data=[[p] for p in p_values], columns=["p_value"]),
            "attacked_lengths": wandb.Table(
                data=[[length] for length in attacked_lengths],
                columns=["attacked_length"],
            ),
        }

        if evaluate_scores:
            negative_texts, _, negative_prompt_token_ids = _generate_samples(
                model=model,
                tokenizer=tokenizer,
                data=data,
                device=device,
                max_length=job.max_length,
                wat=None,
            )
            _, negative_round_trip_texts = _round_trip_texts(
                negative_texts,
                pipeline,
                device,
                batch_size=batch_size,
                max_source_length=max_source_length,
                max_target_length=max_target_length,
            )
            _, neg_scores, _ = _evaluate_attacked_samples(
                wat=wat,
                model=model,
                tokenizer=tokenizer,
                attacked_texts=negative_round_trip_texts,
                prompt_token_ids=negative_prompt_token_ids,
                max_retokenized_length=max_retokenized_length,
                n_samples=n_samples,
                upperbound=upperbound,
            )
            if neg_scores:
                metrics["roc_auc"] = calc_roc_auc_score(pos_scores, neg_scores)
                metrics["tpr_at_fpr"] = calc_tpr_at_fpr(pos_scores, neg_scores, fpr_threshold=0.01)
                tables["pos_scores"] = wandb.Table(
                    data=[[score] for score in pos_scores], columns=["pos_score"]
                )
                tables["neg_scores"] = wandb.Table(
                    data=[[score] for score in neg_scores], columns=["neg_score"]
                )

        wandb_config = {
            "experiment": experiment,
            "dataset_size": dataset_size,
            "model": model.name_or_path,
            "algo": job.wat_name,
            "language": job.language,
            "max_length": job.max_length,
            "translation_forward_model": ROUND_TRIP_MODELS[job.language][0],
            "translation_backward_model": ROUND_TRIP_MODELS[job.language][1],
            "device": str(device),
            "shard_index": shard_index,
            "num_shards": num_shards,
        }
        log_wandb_run(
            experiment=experiment,
            run_name=f"{model.name_or_path}_{job.wat_name}_{job.language}_len{job.max_length}",
            wandb_config=wandb_config,
            metrics=metrics,
            tables=tables,
        )

        _save_examples(
            output_dir=output_dir,
            model_name=model.name_or_path,
            wat_name=job.wat_name,
            language=job.language,
            max_length=job.max_length,
            original_texts=original_texts,
            pivot_texts=pivot_texts,
            round_trip_texts=round_trip_texts,
            attacked_lengths=attacked_lengths,
            n_examples=n_examples,
        )

        print("WandB config:")
        for key, value in wandb_config.items():
            print(f"  {key}: {value}")
        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    for pipeline in pipelines.values():
        del pipeline.forward_model
        del pipeline.backward_model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Completed {experiment}")
