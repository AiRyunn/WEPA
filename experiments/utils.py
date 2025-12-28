import os
import pickle
from itertools import islice

import numpy as np
import torch
import wandb
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermarker.exp import ExpWatermarker
from watermarker.kgw import KGWWatermarker
from watermarker.unbiased import UnbiasedConfig, UnbiasedWatermarker
from watermarker.wepa import WepaWatermarker


def init_wandb(project, name, config):
    run = wandb.init(project=project, name=name, config=config)
    return run


def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Sampling parameters does not work with some watermarkers. Here we disable them.
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    model.to(device)
    print(f"Model: {model.name_or_path}")

    return model, tokenizer


def load_c4(tokenizer, dataset_size, max_length=50):
    # Parameters
    cache_path = "./cache/" + tokenizer.name_or_path + f"_{dataset_size}_{max_length}.pkl"
    nocache = not os.path.exists(cache_path)  # Set to True if no cache exists

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if nocache:
        print("Cache not found. Downloading and caching the dataset...")

        # Load dataset
        dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)

        # Filter dataset
        dataset = dataset.filter(lambda item: len(tokenizer(item["text"]).input_ids) >= 512)

        # Process dataset
        # dataset = dataset.map(
        #     lambda item: {
        #         "input_ids": tokenizer(
        #             item["text"],
        #             add_special_tokens=True,
        #             truncation=True,
        #             max_length=max_length,
        #         )["input_ids"]
        #     }
        # )

        # Limit dataset size
        dataset = islice(dataset, dataset_size)
        data = list(
            map(
                lambda item: tokenizer(
                    item["text"],
                    return_tensors="pt",
                    max_length=max_length,
                ),
                dataset,
            )
        )

        # Save to cache
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
    else:
        print("Loading dataset from cache...")

        # Load cached data
        with open(cache_path, "rb") as f:
            data = pickle.load(f)

    return data


def load_default_kgw_watermarker(vocab_size, device, seed=42):
    return KGWWatermarker(
        key=1,
        vocab_size=vocab_size,
        green_ratio=0.25,
        delta=2,
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
    )


def load_default_unbiased_watermarker(vocab_size, device, seed=42):
    config = UnbiasedConfig(vocab_size=vocab_size, device=device)
    return UnbiasedWatermarker(config)


def load_default_exp_watermarker(vocab_size, device, lam=256, seed=42):
    return ExpWatermarker(
        lam=lam,
        vocab_size=vocab_size,
        block_size=256,
        gamma=0,
        shift=True,
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
    )


def load_default_wepa_watermarker(vocab_size, device, lam=256, degree=1, bits=None, seed=42):
    return WepaWatermarker(
        lam=lam,
        vocab_size=vocab_size,
        degree=degree,
        bits=bits,
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
    )


class UnwatermarkedWatermarker:

    def generate(self, model, inputs, max_new_tokens):
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
        return outputs


def load_unwatermarked_watermarker():
    return UnwatermarkedWatermarker()


def calc_roc_auc_score(pos_scores, neg_scores):
    from sklearn.metrics import roc_auc_score

    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_scores = pos_scores + neg_scores
    roc_auc = roc_auc_score(y_true, y_scores)

    return max(roc_auc, 1 - roc_auc)


def calc_tpr_at_fpr(pos_scores, neg_scores, fpr_threshold=0.01):
    import numpy as np
    from sklearn.metrics import roc_curve

    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_scores = pos_scores + neg_scores
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Find the TPR at the specified FPR threshold
    tpr_at_fpr = 0.0
    for i in range(len(fpr)):
        if fpr[i] <= fpr_threshold:
            tpr_at_fpr = tpr[i]
        else:
            break

    return tpr_at_fpr


def tpr_at_fpr(y_true, y_score, fpr_threshold=0.01):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Sort by descending score
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    # Total positives and negatives
    P = np.sum(y_true_sorted == 1)
    N = np.sum(y_true_sorted == 0)

    if P == 0 or N == 0:
        return 0.0

    # Cumulative TP and FP as we move the threshold
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    # Compute FPR and TPR arrays
    fprs = fps / N
    tprs = tps / P

    # Find the largest index where FPR <= threshold
    idx = np.searchsorted(fprs, fpr_threshold, side="right") - 1
    idx = np.clip(idx, 0, len(tprs) - 1)

    return tprs[idx]


def get_wat_name(wat):
    if isinstance(wat, WepaWatermarker):
        name = "wepa"
        name += f"_d{wat.degree}"
        if wat.bits is not None:
            name += f"_bits{wat.bits}"
        if wat.lam != 256:
            name += f"_lam{wat.lam}"
        return name
    elif isinstance(wat, ExpWatermarker):
        return "exp"
    elif isinstance(wat, KGWWatermarker):
        return "kgw"
    elif isinstance(wat, UnbiasedWatermarker):
        return "unbiased"
    elif isinstance(wat, UnwatermarkedWatermarker):
        return "unwatermarked"

    raise ValueError("Unknown watermarker type")


def run_experiment(
    experiment,
    run_name,
    wat,
    model,
    tokenizer,
    data,
    max_length,
    device,
    evaluate_scores=True,
    upperbound=False,
):
    print(f"Running experiment: {wat}")

    wat_name = get_wat_name(wat)

    p_values = []
    all_token_ids = []
    dataset_size = len(data)
    for i in tqdm(range(dataset_size)):
        for j in range(10):
            inputs = data[i].to(device)
            outputs = wat.generate(
                model,
                inputs,
                max_new_tokens=max_length,
            )
            outputs = outputs[0, inputs["input_ids"].size(1) :]

            if len(outputs) < max_length:
                output_text = tokenizer.decode(outputs, skip_special_tokens=True)
                print(
                    "Text length is less than max_length: ",
                    len(outputs),
                    output_text,
                )
                continue

            # Compute p-value
            if isinstance(wat, UnbiasedWatermarker):
                token_ids = inputs["input_ids"][-wat.config.prefix_length :][0]
                token_ids = torch.cat([token_ids, outputs], dim=0).unsqueeze(0)
                p_value = wat.p_value(token_ids, model=model)
                all_token_ids.append(token_ids)
            else:
                if not upperbound:
                    p_value = wat.p_value(outputs, n_samples=10000)
                else:
                    p_value = wat.p_value(outputs, n_samples=20, upperbound=True)
                all_token_ids.append(outputs)
            p_values.append(p_value)
            break

    p_values_tensor = torch.tensor(p_values)
    p_value_median = float(torch.median(p_values_tensor))

    metrics = {
        "p_value_median": p_value_median,
        "p_value_mean": float(p_values_tensor.mean()),
        "p_value_min": float(p_values_tensor.min()),
        "p_value_max": float(p_values_tensor.max()),
        "n_samples": len(p_values),
    }
    tables = {}
    tables["p_values"] = wandb.Table(data=[[p] for p in p_values], columns=["p_value"])

    print(f"median={p_value_median:.4f} ({len(p_values)} samples)")

    if evaluate_scores:
        pos_scores = []
        neg_scores = []

        for token_ids in all_token_ids:
            if isinstance(wat, UnbiasedWatermarker):
                pos_scores.append(wat.test_statistic(model, token_ids))
            else:
                pos_scores.append(wat.test_statistic(token_ids))

        for i in tqdm(range(dataset_size)):
            for j in range(10):
                inputs = data[i].to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                )
                outputs = outputs[0, inputs["input_ids"].size(1) :]

                if len(outputs) < max_length:
                    output_text = tokenizer.decode(outputs, skip_special_tokens=True)
                    print(
                        "Text length is less than max_length: ",
                        len(outputs),
                        output_text,
                    )
                    continue

                if isinstance(wat, UnbiasedWatermarker):
                    token_ids = inputs["input_ids"][-wat.config.prefix_length :][0]
                    outputs = torch.cat([token_ids, outputs], dim=0).unsqueeze(0)

                if isinstance(wat, UnbiasedWatermarker):
                    neg_scores.append(wat.test_statistic(model, outputs))
                else:
                    neg_scores.append(wat.test_statistic(outputs))
                break

        print("pos_scores:", pos_scores)
        print("neg_scores:", neg_scores)

        roc_auc = calc_roc_auc_score(pos_scores, neg_scores)
        metrics["roc_auc"] = roc_auc
        print(f"ROC AUC: {roc_auc:.4f}")

        tpr_at_fpr = calc_tpr_at_fpr(pos_scores, neg_scores, fpr_threshold=0.01)
        metrics["tpr_at_fpr"] = tpr_at_fpr
        print(f"TPR@1%FPR: {tpr_at_fpr:.4f}")

        tables["pos_scores"] = wandb.Table(data=[[s] for s in pos_scores], columns=["pos_score"])
        tables["neg_scores"] = wandb.Table(data=[[s] for s in neg_scores], columns=["neg_score"])

    run = init_wandb(
        project=f"WEPA-{experiment}",
        name=run_name,
        config={
            "experiment": experiment,
            "dataset_size": dataset_size,
            "model": model.name_or_path,
            "algo": wat_name,
            "text_length": max_length,
            "lam": getattr(wat, "lam", None),
            "bits": getattr(wat, "bits", None),
            "degree": getattr(wat, "degree", None),
            "delta": getattr(wat, "delta", None),
            "device": str(device),
        },
    )

    # Log scalar stats
    wandb.log(metrics)

    # Log tabular data
    for key, value in tables.items():
        wandb.log({key: value})

    run.finish()
