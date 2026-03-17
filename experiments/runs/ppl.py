import torch
from tqdm import tqdm
import wandb

from experiments.utils import (
    build_runtime,
    generate_valid_sample,
    load_default_dip_watermarker,
    load_default_wepa_watermarker,
    load_default_exp_watermarker,
    load_default_kgw_watermarker,
    load_default_unbiased_watermarker,
    get_wat_name,
    load_unwatermarked_watermarker,
    log_wandb_run,
)

experiment = "ppl"

def run(config):
    print(config)

    max_length = config.get("max_length")
    seed = config.get("seed")
    device, model, tokenizer, data = build_runtime(config)
    dataset_size = len(data)

    wats = [
        load_default_wepa_watermarker(
            model.config.vocab_size,
            device,
            degree=1,
            seed=seed,
        ),
        load_default_wepa_watermarker(
            model.config.vocab_size,
            device,
            degree=2,
            seed=seed,
        ),
        load_default_exp_watermarker(
            model.config.vocab_size,
            device,
            seed=seed,
        ),
        load_default_dip_watermarker(
            model.config.vocab_size,
            device,
            seed=seed,
        ),
        load_default_kgw_watermarker(
            model.config.vocab_size,
            device,
            seed=seed,
        ),
        load_default_unbiased_watermarker(
            model.config.vocab_size,
            device,
            seed=seed,
        ),
        load_unwatermarked_watermarker(),
    ]

    def compute_ppl(model, input_ids, continuation_ids, device):
        # concat: [prompt | continuation]
        full_ids = torch.cat([input_ids, continuation_ids.unsqueeze(0)], dim=1).to(device)
        labels = full_ids.clone()
        # mask out the prompt so it's not counted in loss
        labels[:, : input_ids.size(1)] = -100  # -100 is ignored in HF loss

        with torch.no_grad():
            outputs = model(full_ids, labels=labels)
            neg_log_likelihood = outputs.loss.item()
        return torch.exp(torch.tensor(neg_log_likelihood)).item()

    def evaluate_statistics(wat, max_length):
        print(f"max_length: {max_length}")

        ppls = []
        for i in tqdm(range(dataset_size)):
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

            prompt_ids = sample.prompt_token_ids.unsqueeze(0)
            continuation_ids = sample.continuation_token_ids
            ppl = compute_ppl(model, prompt_ids, continuation_ids, device)
            ppls.append(ppl)

        return ppls

    for wat in wats:
        print(f"Running experiment: {wat}")

        wat_name = get_wat_name(wat)
        run_name = f"{model.name_or_path}_{wat_name}"

        ppls = evaluate_statistics(wat, max_length)
        ppl_median = torch.median(torch.tensor(ppls)).item()
        ppl_mean = torch.mean(torch.tensor(ppls)).item()

        metrics = {
            "ppl_median": ppl_median,
            "ppl_mean": ppl_mean,
            "ppl_min": float(torch.min(torch.tensor(ppls))),
            "ppl_max": float(torch.max(torch.tensor(ppls))),
            "n_samples": len(ppls),
        }
        tables = {"ppls": wandb.Table(data=[[p] for p in ppls], columns=["ppl"])}

        print(f"median={ppl_median:.4f} ({len(ppls)} samples)")
        print(f"mean={ppl_mean:.4f} ({len(ppls)} samples)")

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

    print(f"Completed {experiment}")
