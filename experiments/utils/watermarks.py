import torch

from watermarker.dip import DIPConfig, DipWatermarker
from watermarker.exp import ExpWatermarker
from watermarker.kgw import KGWWatermarker
from watermarker.unbiased import UnbiasedConfig, UnbiasedWatermarker
from watermarker.wepa import WepaWatermarker


def load_default_dip_watermarker(vocab_size, device, seed: int = 42):
    seed = int(seed)
    config = DIPConfig(
        vocab_size=vocab_size,
        device=device,
        hash_key=str(seed).encode("utf-8"),
    )
    return DipWatermarker(config)


def load_default_kgw_watermarker(vocab_size, device, seed: int = 42):
    seed = int(seed)
    return KGWWatermarker(
        key=1,
        vocab_size=vocab_size,
        green_ratio=0.25,
        delta=2,
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
    )


def load_default_unbiased_watermarker(vocab_size, device, seed: int = 42):
    seed = int(seed)
    config = UnbiasedConfig(vocab_size=vocab_size, device=device)
    return UnbiasedWatermarker(config)


def load_default_exp_watermarker(vocab_size, device, lam=256, seed: int = 42):
    seed = int(seed)
    return ExpWatermarker(
        lam=lam,
        vocab_size=vocab_size,
        block_size=256,
        gamma=0,
        shift=True,
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
    )


def load_default_wepa_watermarker(
    vocab_size,
    device,
    lam=256,
    degree=1,
    bits=None,
    seed: int = 42,
):
    seed = int(seed)
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


def get_wat_name(wat):
    if isinstance(wat, WepaWatermarker):
        name = "wepa"
        name += f"_d{wat.degree}"
        if wat.bits is not None:
            name += f"_bits{wat.bits}"
        if wat.lam != 256:
            name += f"_lam{wat.lam}"
        return name
    if isinstance(wat, ExpWatermarker):
        return "exp"
    if isinstance(wat, DipWatermarker):
        return "dip"
    if isinstance(wat, KGWWatermarker):
        return "kgw"
    if isinstance(wat, UnbiasedWatermarker):
        return "unbiased"
    if isinstance(wat, UnwatermarkedWatermarker):
        return "unwatermarked"

    raise ValueError("Unknown watermarker type")
