import os
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import islice
import math
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm


def translate_texts(
    texts,
    model,
    tokenizer,
    device,
    batch_size=8,
    max_source_length=256,
    max_target_length=256,
):
    translations = []
    for start in tqdm(range(0, len(texts), batch_size)):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_target_length,
            )
        translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return translations


@dataclass
class GeneratedSample:
    prompt_token_ids: torch.Tensor
    continuation_token_ids: torch.Tensor
    continuation_text: str


def generate_valid_sample(model, tokenizer, inputs, max_length, device, wat=None, max_attempts=10):
    inputs = inputs.to(device)
    for _ in range(max_attempts):
        if wat is None:
            outputs = model.generate(**inputs, max_new_tokens=max_length)
        else:
            outputs = wat.generate(model, inputs, max_new_tokens=max_length)
        continuation_ids = outputs[0, inputs["input_ids"].size(1) :]

        if len(continuation_ids) < max_length:
            output_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
            print(
                "Text length is less than max_length: ",
                len(continuation_ids),
                output_text,
            )
            continue

        return GeneratedSample(
            prompt_token_ids=inputs["input_ids"][0].detach().cpu(),
            continuation_token_ids=continuation_ids.detach().cpu(),
            continuation_text=tokenizer.decode(continuation_ids, skip_special_tokens=True),
        )
    return None


def save_text_results(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".jsonl":
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    else:
        df.to_csv(output_path, index=False)
    return output_path


def count_conditional_ngrams(token_sequences, order):
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}.")

    context_counts = Counter()
    continuation_counts = defaultdict(Counter)
    total_transitions = 0

    for token_sequence in token_sequences:
        if len(token_sequence) <= order:
            continue

        normalized_tokens = [int(token_id) for token_id in token_sequence]
        for index in range(order, len(normalized_tokens)):
            context = tuple(normalized_tokens[index - order : index])
            next_token = normalized_tokens[index]
            context_counts[context] += 1
            continuation_counts[context][next_token] += 1
            total_transitions += 1

    return context_counts, continuation_counts, total_transitions


def estimate_conditional_entropy(token_sequences, order):
    context_counts, continuation_counts, total_transitions = count_conditional_ngrams(
        token_sequences,
        order,
    )
    if total_transitions == 0:
        raise ValueError(
            "Not enough tokens to estimate conditional entropy for "
            f"order={order}. Increase max_length or decrease order."
        )

    entropy_bits = 0.0
    for context, next_token_counts in continuation_counts.items():
        context_total = context_counts[context]
        context_probability = context_total / total_transitions
        conditional_entropy = 0.0
        for count in next_token_counts.values():
            token_probability = count / context_total
            conditional_entropy -= token_probability * math.log2(token_probability)
        entropy_bits += context_probability * conditional_entropy

    return {
        "order": order,
        "conditional_entropy_bits": entropy_bits,
        "total_transitions": total_transitions,
        "n_contexts": len(context_counts),
    }


def load_c4(tokenizer, dataset_size, max_length=50):
    cache_path = (
        "./artifacts/cache/" + tokenizer.name_or_path + f"_{dataset_size}_{max_length}.pkl"
    )
    nocache = not os.path.exists(cache_path)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if nocache:
        print("Cache not found. Downloading and caching the dataset...")
        dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
        dataset = dataset.filter(lambda item: len(tokenizer(item["text"]).input_ids) >= 512)
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
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
    else:
        print("Loading dataset from cache...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)

    return data
