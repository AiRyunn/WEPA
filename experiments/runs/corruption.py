import torch
from tqdm import tqdm
import wandb

from experiments.utils import (
    build_runtime,
    calc_roc_auc_score,
    calc_tpr_at_fpr,
    get_wat_name,
    log_wandb_run,
    load_default_dip_watermarker,
    load_default_exp_watermarker,
    load_default_kgw_watermarker,
    load_default_unbiased_watermarker,
    load_default_wepa_watermarker,
    UnbiasedWatermarker,
)

experiment = "corruption"

def run(config):
    print(config)

    corruption_type = config.get("corruption_type")
    max_length = config.get("max_length")
    evaluate_scores = config.get("evaluate_scores")
    seed = config.get("seed")
    model_name = config.get("model_name")

    if corruption_type not in ["substitution", "deletion", "insertion"]:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

    device, model, tokenizer, data = build_runtime(config)
    dataset_size = len(data)
    corruption_fractions_config = config.get(
        "corruption_fractions",
        {
            "substitution_deletion": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "insertion": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
    )

    if corruption_type in ('substitution', 'deletion'):
        corruption_fractions = corruption_fractions_config["substitution_deletion"]
    elif corruption_type == 'insertion':
        corruption_fractions = corruption_fractions_config["insertion"]
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

    vocab_size = model.config.vocab_size

    def random_substitution(token_ids, fraction):
        n_tokens = int(fraction * len(token_ids))
        indices = torch.randperm(len(token_ids))[:n_tokens]

        # Randomly substitute a fraction of the output sequence
        token_ids = torch.tensor(token_ids)
        token_ids[indices] = torch.randint(0, vocab_size, (n_tokens,), device=device)

        return token_ids

    def random_deletion(token_ids, fraction):
        n_tokens = int(fraction * len(token_ids))
        indices = torch.randperm(len(token_ids), device=device)[:n_tokens]
        mask = torch.ones(len(token_ids), dtype=torch.bool, device=device)
        mask[indices] = 0
        # Randomly delete a fraction of the output sequence
        token_ids = torch.masked_select(torch.tensor(token_ids), mask)

        return token_ids

    def random_insertion(token_ids, fraction):
        n_tokens = int(fraction * len(token_ids))
        random_tokens = torch.randint(0, vocab_size, (n_tokens,))

        # Insert tokens at random positions
        token_ids = torch.tensor(token_ids)
        for i in range(n_tokens):
            index = torch.randint(0, len(token_ids) + 1, (1,), device=device)
            token_ids = torch.cat(
                [
                    token_ids[:index],
                    torch.tensor([random_tokens[i]], device=device),
                    token_ids[index:],
                ]
            )

        return token_ids

    if corruption_type == "substitution":
        corruption_func = random_substitution
    elif corruption_type == "deletion":
        corruption_func = random_deletion
    elif corruption_type == "insertion":
        corruption_func = random_insertion
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

    def run_experiment(wat, corruption_fraction):
        print(f"Running experiment: {wat}")

        wat_name = get_wat_name(wat)

        p_values = []
        all_token_ids = []
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

                outputs = corruption_func(outputs, corruption_fraction)

                # Compute p-value
                if isinstance(wat, UnbiasedWatermarker):
                    token_ids = inputs["input_ids"][-wat.config.prefix_length :][0]
                    token_ids = torch.cat([token_ids, outputs], dim=0).unsqueeze(0)
                    result = wat.p_value(token_ids, model=model)
                    all_token_ids.append(token_ids)
                else:
                    result = wat.p_value(outputs, n_samples=10000)
                    all_token_ids.append(outputs)
                p_values.append(result)
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

        print(
            f"[{wat_name} | corruption_fraction={corruption_fraction}] median={p_value_median:.4f} ({len(p_values)} samples)"
        )

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

                    outputs = corruption_func(outputs, corruption_fraction)

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
            print(f"[{wat_name} | corruption_fraction={corruption_fraction}] ROC AUC: {roc_auc:.4f}")

            tpr_at_fpr = calc_tpr_at_fpr(pos_scores, neg_scores, fpr_threshold=0.01)
            metrics["tpr_at_fpr"] = tpr_at_fpr
            print(
                f"[{wat_name} | corruption_fraction={corruption_fraction}] TPR at FPR=0.01: {tpr_at_fpr:.4f}"
            )

            tables["pos_scores"] = wandb.Table(data=[[s] for s in pos_scores], columns=["pos_score"])
            tables["neg_scores"] = wandb.Table(data=[[s] for s in neg_scores], columns=["neg_score"])

        wandb_config = {
            "experiment": experiment,
            "corruption_type": corruption_type,
            "dataset_size": dataset_size,
            "model": model_name,
            "algo": wat_name,
            "max_length": max_length,
            "corruption_fraction": corruption_fraction,
            "device": str(device),
        }
        log_wandb_run(
            experiment=experiment,
            run_name=f"{model_name}_{wat_name}_len{max_length}",
            wandb_config=wandb_config,
            metrics=metrics,
            tables=tables,
        )

    wats = [
        load_default_wepa_watermarker(
            model.config.vocab_size,
            device,
            degree=1,
        ),
        load_default_wepa_watermarker(
            model.config.vocab_size,
            device,
            degree=2,
        ),
        load_default_exp_watermarker(
            model.config.vocab_size,
            device,
        ),
        load_default_dip_watermarker(
            model.config.vocab_size,
            device,
        ),
        load_default_kgw_watermarker(
            model.config.vocab_size,
            device,
        ),
        load_default_unbiased_watermarker(
            model.config.vocab_size,
            device,
        ),
    ]

    for wat in wats:
        for corruption_fraction in corruption_fractions:
            run_experiment(wat, corruption_fraction)

    print(f"Completed {experiment}")
