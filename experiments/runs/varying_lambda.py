from experiments.utils import (
    build_runtime,
    load_default_wepa_watermarker,
    run_experiment,
    get_wat_name,
    load_default_exp_watermarker,
)

experiment = "varying_lambda"

def run(config):
    print(config)

    max_length = config.get("max_length")
    evaluate_scores = config.get("evaluate_scores")
    degrees = config.get("degrees", [None, 1, 2])
    lams = config.get("lams", [16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    device, model, tokenizer, data = build_runtime(config)

    for degree in degrees:
        for lam in lams:
            if degree is None:
                wat = load_default_exp_watermarker(model.config.vocab_size, device, lam=lam)
            else:
                wat = load_default_wepa_watermarker(
                    model.config.vocab_size, device, degree=degree, lam=lam
                )
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
            )

    print(f"Completed {experiment}")
