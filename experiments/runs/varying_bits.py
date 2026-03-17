from experiments.utils import build_runtime, load_default_wepa_watermarker, run_experiment, get_wat_name

experiment = "varying_bits"

def run(config):
    print(config)

    max_length = config.get("max_length")
    evaluate_scores = config.get("evaluate_scores")
    degrees = config.get("degrees", [1, 2])
    bits_params = config.get("bits_params", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 32])
    device, model, tokenizer, data = build_runtime(config)

    for degree in degrees:
        for bits in bits_params:
            wat = load_default_wepa_watermarker(
                model.config.vocab_size, device, degree=degree, bits=bits
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
