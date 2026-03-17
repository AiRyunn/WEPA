from experiments.utils import (
    build_runtime,
    load_default_dip_watermarker,
    load_default_wepa_watermarker,
    load_default_exp_watermarker,
    load_default_kgw_watermarker,
    load_default_unbiased_watermarker,
    get_wat_name,
    run_experiment,
)

experiment = "varying_length_long"

def run(config):
    print(config)

    evaluate_scores = config.get("evaluate_scores")
    max_lengths = config.get("max_lengths", [25, 50, 75, 100, 125, 150, 175, 200, 225, 250])
    device, model, tokenizer, data = build_runtime(config)

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
    ]
    for wat in wats:
        for max_length in max_lengths:
            wat_name = get_wat_name(wat)

            run_experiment(
                experiment=experiment,
                run_name=f"{model.name_or_path}_{wat_name}",
                wat=wat,
                model=model,
                tokenizer=tokenizer,
                data=data,
                max_length=max_length,
                device=device,
                evaluate_scores=evaluate_scores,
                upperbound=True,
            )

    print(f"Completed {experiment}")
