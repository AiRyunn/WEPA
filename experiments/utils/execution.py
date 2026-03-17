import torch
import wandb
from tqdm import tqdm

from experiments.utils.logging import log_wandb_run
from experiments.utils.metrics import calc_roc_auc_score, calc_tpr_at_fpr
from experiments.utils.watermarks import get_wat_name
from watermarker.unbiased import UnbiasedWatermarker


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
    tables = {"p_values": wandb.Table(data=[[p] for p in p_values], columns=["p_value"])}

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

        metrics["roc_auc"] = calc_roc_auc_score(pos_scores, neg_scores)
        metrics["tpr_at_fpr"] = calc_tpr_at_fpr(pos_scores, neg_scores, fpr_threshold=0.01)

        tables["pos_scores"] = wandb.Table(data=[[s] for s in pos_scores], columns=["pos_score"])
        tables["neg_scores"] = wandb.Table(data=[[s] for s in neg_scores], columns=["neg_score"])

    wandb_config = {
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
    }
    log_wandb_run(
        experiment=experiment,
        run_name=run_name,
        wandb_config=wandb_config,
        metrics=metrics,
        tables=tables,
    )
