# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ===========================================================
# unbiased.py
# Description: Implementation of Unbiased Watermark algorithm
# ===========================================================

import hashlib
import random
from functools import partial
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import LogitsProcessor, LogitsProcessorList


class TransformersConfig:
    """Configuration class for transformers."""

    def __init__(self, model, tokenizer, vocab_size=None, device='cuda', *args, **kwargs):
        """
        Initialize the transformers configuration.

        Parameters:
            model (object): The model object.
            tokenizer (object): The tokenizer object.
            vocab_size (int): The vocabulary size.
            device (str): The device to use.
            kwargs: Additional keyword arguments.
        """
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer) if vocab_size is None else vocab_size


class BaseConfig:
    """Base configuration class for watermark algorithms."""

    def __init__(
        self,
        vocab_size: int,
        device: torch.device = torch.device("cpu"),
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the base configuration.

        Parameters:
            algorithm_config (str): Path to the algorithm configuration file.
            **kwargs: Additional parameters to override config values
        """
        self.config_dict = {
            "algorithm_name": "Unbiased",
            "type": "delta",
            "n_grid": 10,
            "key": 42,
            "prefix_length": 1,
            "p_threshold": 0.0005,
            "ignore_history_generation": 1,
            "ignore_history_detection": 1,
        }

        # Update config with kwargs
        if kwargs:
            self.config_dict.update(kwargs)

        # Load model-related configurations
        self.vocab_size = vocab_size
        self.device = device

        # Initialize algorithm-specific parameters
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters. Should be overridden by subclasses."""
        raise NotImplementedError

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name. Should be overridden by subclasses."""
        raise NotImplementedError


class BaseWatermark:
    def __init__(
        self,
        algorithm_config: str | BaseConfig,
        *args,
        **kwargs,
    ) -> None:
        pass

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        pass

    def p_value(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[tuple, dict]:
        pass


class WatermarkStrategy:
    """Base class for watermark strategies."""

    def from_random(
        self, rng: Union[torch.Generator, list[torch.Generator]], vocab_size: int
    ) -> torch.LongTensor:
        raise NotImplementedError

    def reweight_logits(
        self, shuffle: torch.LongTensor, p_logits: torch.FloatTensor, alpha: float
    ) -> torch.FloatTensor:
        raise NotImplementedError


class DeltaStrategy(WatermarkStrategy):
    """Strategy for delta-reweight."""

    def from_random(
        self, rng: Union[torch.Generator, list[torch.Generator]], vocab_size: int
    ) -> torch.LongTensor:
        if isinstance(rng, list):
            batch_size = len(rng)
            u = torch.stack(
                [torch.rand((), generator=rng[i], device=rng[i].device) for i in range(batch_size)]
            )
        else:
            u = torch.rand((), generator=rng, device=rng.device)
        return u

    def reweight_logits(
        self, u: torch.LongTensor, p_logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Reweight the logits using the u."""
        cumsum = torch.cumsum(F.softmax(p_logits, dim=-1), dim=-1)
        index = torch.searchsorted(cumsum, u[..., None], right=True)
        index = torch.clamp(index, 0, p_logits.shape[-1] - 1)
        modified_logits = torch.where(
            torch.arange(p_logits.shape[-1], device=p_logits.device) == index,
            torch.full_like(p_logits, 0),
            torch.full_like(p_logits, float("-inf")),
        )
        return modified_logits


class GammaStrategy(WatermarkStrategy):
    """Strategy for gamma-reweight."""

    def from_random(
        self, rng: Union[torch.Generator, list[torch.Generator]], vocab_size: int
    ) -> torch.LongTensor:
        """Generate a permutation from the random number generator."""
        if isinstance(rng, list):
            batch_size = len(rng)
            shuffle = torch.stack(
                [
                    torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
                    for i in range(batch_size)
                ]
            )
        else:
            shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
        return shuffle

    def reweight_logits(
        self, shuffle: torch.LongTensor, p_logits: torch.FloatTensor, alpha: float = 0.5
    ) -> torch.FloatTensor:
        """Reweight the logits using the shuffle and alpha."""
        unshuffle = torch.argsort(shuffle, dim=-1)

        s_p_logits = torch.gather(p_logits, -1, shuffle)
        s_log_cumsum = torch.logcumsumexp(s_p_logits, dim=-1)

        # normalize the log_cumsum to force the last element to be 0
        s_log_cumsum = s_log_cumsum - s_log_cumsum[..., -1:]
        s_cumsum = torch.exp(s_log_cumsum)
        s_p = F.softmax(s_p_logits, dim=-1)

        boundary_1 = torch.argmax((s_cumsum > alpha).to(torch.int), dim=-1, keepdim=True)
        p_boundary_1 = torch.gather(s_p, -1, boundary_1)
        portion_in_right_1 = (torch.gather(s_cumsum, -1, boundary_1) - alpha) / p_boundary_1
        portion_in_right_1 = torch.clamp(portion_in_right_1, 0, 1)
        s_all_portion_in_right_1 = (s_cumsum > alpha).type_as(p_logits)
        s_all_portion_in_right_1.scatter_(-1, boundary_1, portion_in_right_1)

        boundary_2 = torch.argmax((s_cumsum > (1 - alpha)).to(torch.int), dim=-1, keepdim=True)
        p_boundary_2 = torch.gather(s_p, -1, boundary_2)
        portion_in_right_2 = (torch.gather(s_cumsum, -1, boundary_2) - (1 - alpha)) / p_boundary_2
        portion_in_right_2 = torch.clamp(portion_in_right_2, 0, 1)
        s_all_portion_in_right_2 = (s_cumsum > (1 - alpha)).type_as(p_logits)
        s_all_portion_in_right_2.scatter_(-1, boundary_2, portion_in_right_2)

        s_all_portion_in_right = s_all_portion_in_right_2 / 2 + s_all_portion_in_right_1 / 2
        s_shift_logits = torch.log(s_all_portion_in_right)
        shift_logits = torch.gather(s_shift_logits, -1, unshuffle)

        return p_logits + shift_logits


class UnbiasedConfig(BaseConfig):
    """Config class for Unbiased watermark algorithm, load config file and initialize parameters."""

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.alpha = 0.5
        self.ignore_history_generation = bool(self.config_dict['ignore_history_generation'])
        self.ignore_history_detection = bool(self.config_dict['ignore_history_detection'])
        self.p_threshold = self.config_dict['p_threshold']
        self.prefix_length = self.config_dict['prefix_length']
        self.type = self.config_dict['type']
        self.hash_key = random.getrandbits(1024).to_bytes(128, "big")
        self.n_grid = self.config_dict['n_grid']

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'Unbiased'


class UnbiasedUtils:
    """Utility class for Unbiased watermark algorithm, contains helper functions."""

    def __init__(self, config: UnbiasedConfig, *args, **kwargs) -> None:
        """
        Initialize the Unbiased utility class.

        Parameters:
            config (UnbiasedConfig): Configuration for the unbiased algorithm.
        """
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.cc_history = set()
        self.state_indicator = 0  # 0 for generation, 1 for detection and visualization
        self.strategy = {"delta": DeltaStrategy(), "gamma": GammaStrategy()}[self.config.type]

    def _get_rng_seed(self, context_code: any) -> int:
        """Get the random seed from the given context code and private key."""
        if (not self.config.ignore_history_generation and self.state_indicator == 0) or (
            not self.config.ignore_history_detection and self.state_indicator == 1
        ):
            self.cc_history.add(context_code)

        m = hashlib.sha256()
        m.update(context_code)
        m.update(self.config.hash_key)

        full_hash = m.digest()
        seed = int.from_bytes(full_hash, "big") % (2**32 - 1)
        return seed

    def _extract_context_code(self, context: torch.LongTensor) -> bytes:
        """Extract context code from the given context."""
        if self.config.prefix_length == 0:
            return context.detach().cpu().numpy().tobytes()
        else:
            return context[-self.config.prefix_length :].detach().cpu().numpy().tobytes()

    def get_seed_for_cipher(
        self, input_ids: torch.LongTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Get the mask and seeds for the cipher."""
        batch_size = input_ids.size(0)
        context_codes = [self._extract_context_code(input_ids[i]) for i in range(batch_size)]

        mask, seeds = zip(
            *[
                (context_code in self.cc_history, self._get_rng_seed(context_code))
                for context_code in context_codes
            ]
        )

        return mask, seeds

    def _apply_watermark(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Apply watermark to the scores."""
        mask, seeds = self.get_seed_for_cipher(input_ids)

        rng = [torch.Generator(device=scores.device).manual_seed(seed) for seed in seeds]
        mask = torch.tensor(mask, device=scores.device)
        shuffle = self.strategy.from_random(rng, scores.size(1))

        reweighted_scores = self.strategy.reweight_logits(shuffle, scores)

        return mask, reweighted_scores

    def _safe_minus(
        self, q_logits: torch.FloatTensor, p_logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Safe minus operation to avoid numerical instability."""
        llr = q_logits - p_logits
        llr.nan_to_num_(nan=0.0)
        return llr

    def _from_grid(self, dist_ps: npt.ArrayLike, dist_qs: npt.ArrayLike):
        """Generate batch query from grid search."""
        dist_ps = np.array(dist_ps)
        dist_qs = np.array(dist_qs)
        assert dist_ps.ndim == 1
        assert dist_qs.ndim == 1
        assert np.all(dist_ps >= 0) and np.all(dist_qs >= 0)
        with np.errstate(divide="ignore"):
            dist_p_logs = np.log(dist_ps)
            dist_q_logs = np.log(dist_qs)
        dist_p_logs.sort()
        dist_q_logs.sort()
        batch_query = [(d_p_l, d_q_l) for d_p_l in dist_p_logs for d_q_l in dist_q_logs]
        return batch_query

    def _get_max_llr(
        self,  # shape = (..., vocab_size)
        p_logits: torch.FloatTensor,
        q_logits: torch.FloatTensor,
        batch_query: list[tuple[float, float]],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Get the max llr"""
        # shape = (..., vocab_size)
        llr = self._safe_minus(q_logits, p_logits)
        # shape = (..., vocab_size)
        try:
            sort_index = torch.argsort(llr, dim=-1, descending=True)
        except torch.cuda.OutOfMemoryError as e:
            # use cpu instead
            sort_index = torch.argsort(llr.cpu(), dim=-1, descending=True).to(llr.device)
        del llr

        p_logits = p_logits.gather(-1, sort_index)
        q_logits = q_logits.gather(-1, sort_index)
        del sort_index

        # shape = (..., vocab_size)
        llr = self._safe_minus(q_logits, p_logits)

        # shape = (..., vocab_size)
        sum_q_logits = torch.logcumsumexp(q_logits, dim=-1)
        sum_p_logits = torch.logcumsumexp(p_logits, dim=-1)
        del q_logits
        del p_logits

        max_llrs = []
        for dist_p_log, dist_q_log in batch_query:
            # shape = (..., vocab_size)
            modified_q_logits = torch.where(
                sum_q_logits <= dist_q_log,
                torch.tensor(float("-inf"), device=sum_q_logits.device, dtype=sum_q_logits.dtype),
                sum_q_logits + torch.log(-torch.expm1(dist_q_log - sum_q_logits)),
            )
            modified_p_logits = torch.logaddexp(
                sum_p_logits,
                torch.tensor(dist_p_log, device=sum_p_logits.device, dtype=sum_p_logits.dtype),
            )

            # shape = (..., vocab_size)
            modified_llr = self._safe_minus(modified_q_logits, modified_p_logits)
            del modified_p_logits
            del modified_q_logits

            # pad left modified_llr with -inf
            # shape = (..., vocab_size+1)
            modified_llr = F.pad(modified_llr, (1, 0), value=float("-inf"))
            # get index by argmax
            # shape = (..., )
            cut_index = torch.where(
                torch.any(llr < modified_llr[..., :-1], dim=-1),
                torch.argmax((llr < modified_llr[..., :-1]).to(torch.int), dim=-1),
                torch.tensor(modified_llr.shape[-1] - 1, device=modified_llr.device),
            )
            # shape = (..., 1)
            max_llrs.append(modified_llr.gather(-1, cut_index.unsqueeze(-1)))
        # shape = (..., query_size)
        max_llr = torch.cat(max_llrs, dim=-1)
        del max_llrs
        return max_llr

    @torch.no_grad()
    def _score_llr(
        self,
        p_logits: torch.FloatTensor,
        q_logits: torch.FloatTensor,
        batch_query: list[tuple[float, float]],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        return (llr, max_llr, min_llr)
        llr: [batch_size, seq_len, vocab_size]
        max_llr: [batch_size, seq_len, query_size]
        min_llr: [batch_size, seq_len, query_size]
        """
        q_logits = F.log_softmax(q_logits, dim=-1)
        p_logits = F.log_softmax(p_logits, dim=-1)

        max_llr = self._get_max_llr(p_logits, q_logits, batch_query)
        min_llr = -self._get_max_llr(q_logits, p_logits, [(q, p) for p, q in batch_query])
        trivial_pos = max_llr < min_llr
        max_llr = torch.where(trivial_pos, torch.tensor(0.0, device=max_llr.device), max_llr)
        min_llr = torch.where(trivial_pos, torch.tensor(0.0, device=min_llr.device), min_llr)

        llr = self._safe_minus(q_logits, p_logits)
        return llr, max_llr, min_llr

    def score_sequence(self, model: torch.nn.Module, input_ids: torch.Tensor) -> float:
        """Score the input_ids and return z_score and green_token_flags."""
        # Step 1: Get dist for grid-search
        n = self.config.n_grid
        dist = [float(i) / n for i in range(n + 1)]
        batch_query = self._from_grid([0.0], dist)

        # Step 2: Get the original logits & modified logits
        ## Ignore new token logits
        # attention_mask = inputs["attention_mask"][..., :-1].to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)
        # Ignore prefix token's score
        labels = input_ids[..., self.config.prefix_length :].to(model.device)
        input_ids = input_ids[..., :-1].to(model.device)

        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits
        old_logits = torch.clone(logits)
        new_logits = torch.clone(logits)

        for i in range(
            self.config.prefix_length - 1, logits.size(1) - 1
        ):  # Skip prefix tokens & logits for new token
            pre_input_ids = input_ids[:, : i + 1]
            # Original logits
            t = logits[:, i]
            # Modified logits
            mask, reweighted_scores = self._apply_watermark(pre_input_ids, t)

            old_logits[:, i] = t
            if self.config.ignore_history_detection:
                new_logits[:, i] = reweighted_scores
            else:
                new_logits[:, i] = torch.where(mask[:, None], t, reweighted_scores)

        old_logits = old_logits[:, self.config.prefix_length - 1 :]
        new_logits = new_logits[:, self.config.prefix_length - 1 :]

        # Step 3: Get llr, max_llr, min_llr
        llr, max_llr, min_llr = self._score_llr(old_logits, new_logits, batch_query)

        # Step 4: Extract & Clamp llr for text's token ids
        unclipped_scores = torch.gather(llr, -1, labels.unsqueeze(-1)).squeeze(-1)
        ## shape = (batch_size, seq_len - prefix_length, query_size)
        scores = torch.clamp(unclipped_scores.unsqueeze(-1), min_llr, max_llr)
        scores = np.array(scores[0].cpu())
        labels = np.array(labels[0].cpu())

        # Step 5: Choose the best index for grid search
        sum_scores = np.sum(scores, axis=0)
        best_index = np.argmax(sum_scores)
        final_score = sum_scores[best_index]

        ## Theorem 9: p <= A * e^(-t)
        p_val = n * np.exp(-final_score)

        return p_val


class UnbiasedLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for DiP algorithm, process logits to add watermark."""

    def __init__(self, config: UnbiasedConfig, utils: UnbiasedUtils, *args, **kwargs) -> None:
        """
        Initialize the Unbiased logits processor.

        Parameters:
            config (UnbiasedConfig): Configuration for the Unbiased algorithm.
            utils (UnbiasedUtils): Utility class for the Unbiased algorithm.
        """
        self.config = config
        self.utils = utils

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        mask, reweighted_scores = self.utils._apply_watermark(input_ids, scores)

        if self.config.ignore_history_generation:
            return reweighted_scores
        else:
            return torch.where(mask[:, None], scores, reweighted_scores)


class UnbiasedWatermarker(BaseWatermark):
    """Top-level class for Unbiased algorithm."""

    def __init__(
        self,
        config: UnbiasedConfig,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the Unbiased algorithm.

        Parameters:
            algorithm_config (str | UnbiasedConfig): Path to the algorithm configuration file or UnbiasedConfig instance.
        """
        self.config = config

        self.utils = UnbiasedUtils(self.config)
        self.logits_processor = UnbiasedLogitsProcessor(self.config, self.utils)

    def generate(
        self,
        model: torch.nn.Module,
        inputs: dict,
        max_new_tokens: int,
        *args,
        **kwargs,
    ):
        """Generate watermarked text."""

        # Set the state indicator to 0 for generation
        self.utils.state_indicator = 0

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            max_new_tokens=max_new_tokens,
        )

        # Generate watermarked text
        outputs = generate_with_watermark(**inputs)

        return outputs

    def p_value(self, token_ids: dict, *args, **kwargs):
        """Detect watermark in the text."""

        # Set the state indicator to 1 for detection
        self.utils.state_indicator = 1

        # Compute z-score using a utility method
        p_val = self.utils.score_sequence(kwargs["model"], token_ids)

        # Clear the history
        self.utils.cc_history.clear()

        return p_val

    def scores(
        self,
        model,
        neg_input_ids_list,
        pos_input_ids_list,
    ):
        """
        Compute TPR at a given FPR and AUROC for watermark detection.
        Assumes lower scores = more watermarked.

        Args:
            model: Model used for scoring.
            neg_input_ids_list (list[dict]): Tokenized negatives (no watermark).
            pos_input_ids_list (list[dict]): Tokenized positives (with watermark).
            fpr_target (float): Target false positive rate.

        Returns:
            tuple: (TPR@FPR_target, AUROC)
        """

        # Compute scores for negatives and positives
        neg_scores = [
            self.utils.score_sequence(model, token_ids) for token_ids in neg_input_ids_list
        ]
        pos_scores = [
            self.utils.score_sequence(model, token_ids) for token_ids in pos_input_ids_list
        ]

        neg_scores = np.array(neg_scores)
        pos_scores = np.array(pos_scores)

        all_scores = np.concatenate([neg_scores, pos_scores])
        labels = np.concatenate(
            [
                np.ones(len(neg_scores)),  # negatives = 0
                np.zeros(len(pos_scores)),  # positives = 1
            ]
        )
        return all_scores, labels


    def test_statistic(self, model, input_ids):
        return self.utils.score_sequence(model, input_ids)
