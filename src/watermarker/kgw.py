import math
from functools import cache

import numpy as np
import torch
from matplotlib import defaultParams
from scipy.stats import binom
from sklearn.metrics import roc_auc_score
from torch.distributions import Categorical
from transformers import DynamicCache, LogitsProcessor


class KGWLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        key: int,
        vocab_size: int,
        green_ratio: float = 0.25,
        delta: float = 2,
        gram_size: int = 1,
        generator: torch.Generator | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the Red-Green Set watermark logits processor.

        Args:
            key (int): Key for the watermark.
            vocab_size (int): Size of the vocabulary.
            green_ratio (float): Ratio of green tokens to total tokens.
            delta (float): Bias value for the logits.
            gram_size (int): Size of the n-gram for the watermark.
        """
        self.key = key
        self.vocab_size = vocab_size
        self.green_ratio = green_ratio
        self.delta = delta
        self.gram_size = gram_size
        self.device = device
        if generator is not None:
            self.rng = generator
        else:
            self.rng = torch.Generator(device=device)

        if self.gram_size != 1:
            raise NotImplementedError("Only gram_size=1 is supported for now.")

    def _random_permutation(self, input_ids: torch.Tensor) -> torch.Tensor:
        prev_token = input_ids[-1]
        self.rng.manual_seed(self.key * prev_token.item())
        return torch.randperm(self.vocab_size, generator=self.rng)

    def _get_greenset_ids_cached(self, input_ids: tuple) -> torch.Tensor:
        """
        Get the green set of token IDs based on the input IDs.

        Args:
            input_ids (torch.Tensor): The input token IDs.

        Returns:
            torch.Tensor: The green set of token IDs.
        """
        perm = self._random_permutation(torch.tensor(input_ids, device=self.device))
        split_index = int(self.green_ratio * len(perm))
        return perm[:split_index]

    @cache
    def _get_greenset_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get the green set of token IDs based on the input IDs.

        Args:
            input_ids (torch.Tensor): The input token IDs.

        Returns:
            torch.Tensor: The green set of token IDs.
        """
        greenset_ids = self._get_greenset_ids_cached(tuple(input_ids.tolist()))
        return greenset_ids.to(self.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.gram_size:
            return scores

        batched_greenset_ids = [None for _ in range(input_ids.shape[0])]

        for i, greenset_ids in enumerate(batched_greenset_ids):
            greenset_ids = self._get_greenset_ids(input_ids[i])
            scores[i, greenset_ids] += self.delta

        return scores


class KGWWatermarker:
    def __init__(
        self,
        key: int,
        vocab_size: int,
        green_ratio: float = 0.25,
        delta: float = 2,
        generator: torch.Generator | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the Red-Green Set Watermark.

        Args:
            vocab_size (int): Size of the vocabulary.
            threshold (float): Threshold for dividing the red and green lists.
            seed (int): Random seed for reproducibility.
        """
        self.key = key
        self.vocab_size = vocab_size
        self.green_ratio = green_ratio
        self.gen = generator if generator is not None else torch.Generator()
        self.device = device
        self.delta = delta
        self.logits_processor = KGWLogitsProcessor(
            key=key,
            vocab_size=vocab_size,
            green_ratio=green_ratio,
            delta=delta,
        )

    def _random_permutation(self, input_ids: torch.Tensor) -> torch.Tensor:
        prev_token = input_ids[-1]
        rng = torch.Generator()
        rng.manual_seed(int(self.key * prev_token.item()))
        return torch.randperm(self.vocab_size, generator=rng)

    @cache
    def _get_green_ids_cached(self, input_ids: tuple) -> set:
        """
        Get the green set of token IDs based on the input IDs.

        Args:
            input_ids (tuple): The input token IDs.

        Returns:
            torch.Tensor: The green set of token IDs.
        """
        perm = self._random_permutation(torch.tensor(input_ids, device=self.device))
        split_index = int(self.green_ratio * len(perm))
        return set(perm[:split_index].tolist())

    def _get_green_ids_set(self, input_ids: torch.Tensor) -> set:
        """
        Get the green set of token IDs based on the input IDs.

        Args:
            input_ids (torch.Tensor): The input token IDs.

        Returns:
            torch.Tensor: The green set of token IDs.
        """
        return self._get_green_ids_cached(tuple(input_ids.tolist()))

    def _sample_token(self, logits: torch.Tensor, prev_ids: torch.Tensor) -> torch.Tensor:
        """
        Sample a token from the probability distribution.

        Args:
            logits (torch.Tensor): The logit values of the next token.
            prev_ids (torch.Tensor): The previous token.

        Returns:
            torch.Tensor: The sampled token.
        """
        # Get the green set of token IDs
        green_ids = self._get_green_ids_set(prev_ids)

        # Normalize probabilities for the green set
        logits[0, green_ids] += self.delta

        # Perform Categorical sampling
        return Categorical(logits=logits).sample((1,))

    @torch.no_grad()
    def generate(
        self,
        model: torch.nn.Module,
        inputs: dict,
        max_new_tokens: int,
        # **kwargs,
    ):
        """
        Generate the watermark tokens.

        Args:
            model (torch.nn.Module): The model to generate the watermark tokens.
            input_ids (torch.Tensor): The input sequence to generate the watermark tokens.
            max_length (int): The maximum length of the watermark tokens.
            return_entropy (bool): Whether to return the entropy of the watermark tokens.

        Returns:
            torch.Tensor: The watermark tokens.
        """
        model.eval()

        # Initialize past_key_values for caching
        return model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            logits_processor=[self.logits_processor],
        )

    def test_statistic(self, token_ids: torch.Tensor) -> float:
        """
        Compute the test statistic.

        Args:
            token_ids (torch.Tensor): Sequence of token indices.

        Returns:
            float: Test statistic score.
        """
        dist = 0
        for i in range(1, len(token_ids)):
            greenset_ids = self._get_green_ids_set(token_ids[i - 1 : i])
            if token_ids[i].item() not in greenset_ids:
                dist += 1
        return dist

    def p_value(self, token_ids: torch.Tensor, **kwargs) -> float:
        """
        Compute the p-value of the test statistic.

        Args:
            token_ids (torch.Tensor): Sequence of token indices.
            sample_size (int): Number of samples to estimate the mean and standard deviation.

        Returns:
            float: p-value of the test statistic.
        """
        n = len(token_ids) - 1
        sg = self.test_statistic(token_ids)

        return binom.cdf(sg, n - 1, 1 - self.green_ratio).item()

    def z_score(self, token_ids: torch.Tensor, **kwargs) -> float:
        """
        Compute the z-score of the test statistic.

        Args:
            token_ids (torch.Tensor): Sequence of token indices.

        Returns:
            float: z-score of the test statistic.
        """
        n = len(token_ids)
        sg = n - 1 - self.test_statistic(token_ids)
        mu = (n - 1) * self.green_ratio
        sigma = math.sqrt((n - 1) * self.green_ratio * (1 - self.green_ratio))

        return -(sg - mu) / sigma

    def scores(
        self,
        neg_input_ids_list: list[torch.Tensor],
        pos_input_ids_list: list[torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute TPR at the given FPR using the test statistic.

        Args:
            neg_input_ids_list (list[torch.Tensor]): Negative examples (no watermark).
            pos_input_ids_list (list[torch.Tensor]): Positive examples (with watermark).
            fpr_target (float): Target false positive rate.

        Returns:
            float: True Positive Rate at the specified FPR.
        """

        neg_scores = []
        for token_ids in neg_input_ids_list:
            psi = self.test_statistic(token_ids)
            neg_scores.append(psi)
        neg_scores = np.array(neg_scores)

        pos_scores = []
        for token_ids in pos_input_ids_list:
            psi = self.test_statistic(token_ids)
            pos_scores.append(psi)
        pos_scores = np.array(pos_scores)

        all_scores = np.concatenate([neg_scores, pos_scores])
        labels = np.concatenate(
            [
                np.ones(len(neg_scores)),  # negatives = 0
                np.zeros(len(pos_scores)),  # positives = 1
            ]
        )
        return all_scores, labels
