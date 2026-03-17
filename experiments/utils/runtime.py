import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from experiments.utils.data import load_c4


def apply_seed(seed: int) -> None:
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_index):
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def _load_pretrained_model_and_tokenizer(model_class, model_name, device, dtype, log_prefix):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    print(f"{log_prefix}: {model.name_or_path}")
    return model, tokenizer


def load_model_and_tokenizer(model_name, device):
    model, tokenizer = _load_pretrained_model_and_tokenizer(
        AutoModelForCausalLM,
        model_name,
        device,
        dtype=torch.float16,
        log_prefix="Model",
    )
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Sampling parameters does not work with some watermarkers. Here we disable them.
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    return model, tokenizer


def load_seq2seq_model_and_tokenizer(model_name, device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = _load_pretrained_model_and_tokenizer(
        AutoModelForSeq2SeqLM,
        model_name,
        device,
        dtype=dtype,
        log_prefix="Translation model",
    )
    model.eval()
    return model, tokenizer


def build_runtime(config, dataset_max_length=50):
    seed = config.get("seed", 42)
    apply_seed(seed)

    model_name = config.get("model_name")
    if model_name is None:
        raise ValueError("Config must include 'model_name'.")

    dataset_size = config.get("dataset_size")
    if dataset_size is None:
        raise ValueError("Config must include 'dataset_size'.")

    device = resolve_device(config.get("device", "0"))
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    data = load_c4(tokenizer, dataset_size, max_length=dataset_max_length)
    print(data[0])
    return device, model, tokenizer, data
