import hashlib
import math
from functools import partial

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList


class DIPConfig:
    def __init__(
        self,
        vocab_size: int,
        device: torch.device = torch.device("cpu"),
        alpha: float = 0.5,
        gamma: float = 0.5,
        hash_key: bytes | str | int = b"42",
        prefix_length: int = 5,
        ignore_history_generation: bool = False,
        ignore_history_detection: bool = False,
        z_threshold: float = 4.0,
    ) -> None:
        self.vocab_size = vocab_size
        self.device = device
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.prefix_length = int(prefix_length)
        self.ignore_history_generation = bool(ignore_history_generation)
        self.ignore_history_detection = bool(ignore_history_detection)
        self.z_threshold = float(z_threshold)

        if isinstance(hash_key, bytes):
            self.hash_key = hash_key
        else:
            self.hash_key = str(hash_key).encode("utf-8")


class DIPUtils:
    def __init__(self, config: DIPConfig) -> None:
        self.config = config
        self.cc_history = set()
        self.state_indicator = 0  # 0 for generation, 1 for detection

    def _get_rng_seed(self, context_code: bytes) -> int:
        if (
            (not self.config.ignore_history_generation and self.state_indicator == 0)
            or (not self.config.ignore_history_detection and self.state_indicator == 1)
        ):
            self.cc_history.add(context_code)

        hasher = hashlib.sha256()
        hasher.update(context_code)
        hasher.update(self.config.hash_key)
        full_hash = hasher.digest()
        return int.from_bytes(full_hash, "big") % (2**32 - 1)

    def _extract_context_code(self, context: torch.LongTensor) -> bytes:
        if self.config.prefix_length == 0:
            return context.detach().cpu().numpy().tobytes()
        return context[-self.config.prefix_length :].detach().cpu().numpy().tobytes()

    def from_random(
        self,
        rng: torch.Generator | list[torch.Generator],
        vocab_size: int,
        device: torch.device,
    ) -> torch.LongTensor:
        if isinstance(rng, list):
            return torch.stack(
                [torch.randperm(vocab_size, generator=generator, device=device) for generator in rng]
            )
        return torch.randperm(vocab_size, generator=rng, device=device)

    def reweight_logits(
        self,
        shuffle: torch.LongTensor,
        p_logits: torch.FloatTensor,
    ) -> torch.FloatTensor:
        unshuffle = torch.argsort(shuffle, dim=-1)

        s_p_logits = torch.gather(p_logits, -1, shuffle)
        s_log_cumsum = torch.logcumsumexp(s_p_logits, dim=-1)
        s_log_cumsum = s_log_cumsum - s_log_cumsum[..., -1:]
        s_cumsum = torch.exp(s_log_cumsum)
        s_p = F.softmax(s_p_logits, dim=-1)

        boundary_1 = torch.argmax(
            (s_cumsum > self.config.alpha).to(torch.int64),
            dim=-1,
            keepdim=True,
        )
        p_boundary_1 = torch.gather(s_p, -1, boundary_1)
        portion_in_right_1 = (
            torch.gather(s_cumsum, -1, boundary_1) - self.config.alpha
        ) / p_boundary_1
        portion_in_right_1 = torch.clamp(portion_in_right_1, 0, 1)
        s_all_portion_in_right_1 = (s_cumsum > self.config.alpha).type_as(p_logits)
        s_all_portion_in_right_1.scatter_(-1, boundary_1, portion_in_right_1)

        boundary_2 = torch.argmax(
            (s_cumsum > (1 - self.config.alpha)).to(torch.int64),
            dim=-1,
            keepdim=True,
        )
        p_boundary_2 = torch.gather(s_p, -1, boundary_2)
        portion_in_right_2 = (
            torch.gather(s_cumsum, -1, boundary_2) - (1 - self.config.alpha)
        ) / p_boundary_2
        portion_in_right_2 = torch.clamp(portion_in_right_2, 0, 1)
        s_all_portion_in_right_2 = (s_cumsum > (1 - self.config.alpha)).type_as(p_logits)
        s_all_portion_in_right_2.scatter_(-1, boundary_2, portion_in_right_2)

        s_all_portion_in_right = s_all_portion_in_right_2 / 2 + s_all_portion_in_right_1 / 2
        shift_logits = torch.log(torch.clamp(s_all_portion_in_right, min=1e-20))
        return p_logits + torch.gather(shift_logits, -1, unshuffle)

    def get_seed_for_cipher(self, input_ids: torch.LongTensor) -> tuple[list[bool], list[int]]:
        context_codes = [self._extract_context_code(input_ids[i]) for i in range(input_ids.size(0))]
        mask_and_seeds = [
            (context_code in self.cc_history, self._get_rng_seed(context_code))
            for context_code in context_codes
        ]
        mask, seeds = zip(*mask_and_seeds)
        return list(mask), list(seeds)

    def _get_green_token_quantile(
        self,
        input_ids: torch.LongTensor,
        current_token: torch.LongTensor,
    ) -> tuple[torch.Tensor, bool]:
        mask, seeds = self.get_seed_for_cipher(input_ids.unsqueeze(0))
        rng = [torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds]
        shuffle = self.from_random(rng, self.config.vocab_size, input_ids.device)
        token_quantile = (torch.where(shuffle[0] == current_token)[0] + 1) / self.config.vocab_size
        return token_quantile.reshape(-1)[0], mask[0]

    def _get_dip_score(self, input_ids: torch.LongTensor) -> torch.Tensor:
        scores = torch.zeros(input_ids.shape, dtype=torch.float32, device=input_ids.device)

        for i in range(input_ids.shape[-1] - 1):
            prefix = input_ids[: i + 1]
            current_token = input_ids[i + 1]
            token_quantile, masked = self._get_green_token_quantile(prefix, current_token)
            if not self.config.ignore_history_detection and masked:
                scores[i + 1] = -1
            else:
                scores[i + 1] = token_quantile

        return scores

    def score_sequence(self, input_ids: torch.LongTensor) -> tuple[float, list[int]]:
        scores = self._get_dip_score(input_ids)
        green_token_flags = torch.zeros_like(scores, dtype=torch.long)

        valid_mask = scores >= 0
        green_mask = scores >= self.config.gamma
        green_token_flags[green_mask] = 1
        green_token_flags[: self.config.prefix_length] = -1
        green_token_flags[~valid_mask] = -1

        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return 0.0, green_token_flags.tolist()

        green_tokens = int(green_mask.sum().item())
        expected_green = (1 - self.config.gamma) * valid_count
        z_score = (green_tokens - expected_green) / math.sqrt(valid_count)
        return float(z_score), green_token_flags.tolist()


class DIPLogitsProcessor(LogitsProcessor):
    def __init__(self, config: DIPConfig, utils: DIPUtils) -> None:
        self.config = config
        self.utils = utils

    def _apply_watermark(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> tuple[torch.Tensor, torch.FloatTensor]:
        mask, seeds = self.utils.get_seed_for_cipher(input_ids)
        rng = [torch.Generator(device=scores.device).manual_seed(seed) for seed in seeds]
        shuffle = self.utils.from_random(rng, scores.size(1), scores.device)
        reweighted_scores = self.utils.reweight_logits(shuffle, scores)
        return torch.tensor(mask, device=scores.device), reweighted_scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        mask, reweighted_scores = self._apply_watermark(input_ids, scores)
        if self.config.ignore_history_generation:
            return reweighted_scores
        return torch.where(mask[:, None], scores, reweighted_scores)


class DipWatermarker:
    def __init__(self, config: DIPConfig) -> None:
        self.config = config
        self.utils = DIPUtils(config)
        self.logits_processor = DIPLogitsProcessor(config, self.utils)

    @torch.no_grad()
    def generate(
        self,
        model: torch.nn.Module,
        inputs: dict,
        max_new_tokens: int,
    ) -> torch.Tensor:
        self.utils.state_indicator = 0
        generate_with_watermark = partial(
            model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            max_new_tokens=max_new_tokens,
        )
        outputs = generate_with_watermark(**inputs)
        self.utils.cc_history.clear()
        return outputs

    def test_statistic(self, token_ids: torch.Tensor) -> float:
        self.utils.state_indicator = 1
        z_score, _ = self.utils.score_sequence(token_ids.to(self.config.device))
        self.utils.cc_history.clear()
        return z_score

    def p_value(self, token_ids: torch.Tensor, **kwargs) -> float:
        z_score = self.test_statistic(token_ids)
        return 0.5 * math.erfc(z_score / math.sqrt(2))


class DIP(DipWatermarker):
    pass
