import wandb


def init_wandb(project, name, config):
    run = wandb.init(project=project, name=name, config=config, dir="artifacts/wandb")
    return run


def log_wandb_run(experiment, run_name, wandb_config, metrics, tables=None):
    run = init_wandb(
        project=f"WEPA-{experiment}",
        name=run_name,
        config=wandb_config,
    )
    wandb.log(metrics)
    for key, value in (tables or {}).items():
        wandb.log({key: value})
    run.finish()

    print("WandB config:")
    for key, value in wandb_config.items():
        print(f"  {key}: {value}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
