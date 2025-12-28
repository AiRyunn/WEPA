import math

import numba
import numpy as np
import torch

from watermarker.wepa import WepaWatermarker


@numba.njit(fastmath=True, cache=True)
def _distance_edit_jit(token_ids: np.ndarray, cost: np.ndarray, gamma: float) -> float:
    lam = cost.shape[0]
    length = token_ids.shape[0]

    f0 = np.full(lam + 1, np.inf)
    f1 = np.full(lam + 1, np.inf)

    for j in range(lam):
        f0[j] = j * gamma

    for i in range(1, length + 1):
        f1[0] = f0[0] + gamma
        for j in range(1, lam + 1):
            d0 = cost[j - 1, token_ids[i - 1]]
            f1[j] = min(f0[j] + gamma, f1[j - 1] + gamma, f0[j - 1] + d0)
        f0, f1 = f1, f0

    return f0[lam]


class ExpWatermarker:
    def __init__(
        self,
        lam: int,
        vocab_size: int,
        block_size: int | None = None,
        gamma: float = 0,
        shift: bool = True,
        generator: torch.Generator | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the CyclicKeySampler with the necessary parameters.

        Args:
            lam (int): Length of the key sequence.
            vocab_size (int): Size of the vocabulary.
            block_size (int): Length of the block. Defaults to lam.
            gamma (float): Reduction factor. Defaults to 1.
            shift (bool): Whether to shift the key sequence during decoding. Defaults to True.
            generator (torch.Generator): Random number generator. Defaults to None.
            device (torch.device): Device to store the watermark. Defaults to "cpu".
        """
        self.wepa = WepaWatermarker(
            lam=lam, vocab_size=vocab_size, generator=generator, device=device
        )

        self.lam = lam
        self.vocab_size = vocab_size
        self.block_size = block_size if block_size is not None else lam
        self.gamma = gamma
        self.shift = shift
        self.gen = generator if generator is not None else torch.Generator()
        self.device = device

        if gamma < 0:
            raise ValueError("Gamma must be non-negative")

        self.mu = self.wepa.substates
        self.cost = self.wepa.cost

    @torch.no_grad()
    def generate(
        self,
        model: torch.nn.Module,
        inputs: dict,
        max_new_tokens: int,
    ):
        """
        Generate samples using the CyclicKeySampler with caching.

        Args:
            input_ids (torch.Tensor): Input sequence of token indices.
            max_length (int): Maximum length of the generated sequence.

        Returns:
            torch.Tensor: Generated sequence of token indices.
        """

        return self.wepa.generate(
            model=model,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
        )

    def test_statistic(self, token_ids: torch.Tensor) -> float:
        """
        Compute the test statistic.

        Args:
            token_ids (torch.Tensor): Sequence of token indices.

        Returns:
            float: Test statistic score.
        """
        token_ids_np = token_ids.cpu().numpy()

        lam = self.cost.shape[0]

        block_size = min(self.block_size, len(token_ids_np))
        distances = np.zeros((len(token_ids_np) - block_size + 1, lam))

        for i in range(len(token_ids_np) - block_size + 1):
            for j in range(lam):
                cost_shifted = self.cost[j : j + block_size, :]
                distance = _distance_edit_jit(
                    token_ids_np[i : i + block_size], cost_shifted, self.gamma
                )
                distances[i, j] = distance

        return np.min(distances).item()

    def z_score(self, token_ids: torch.Tensor, n_samples: int = 20) -> float:
        """
        Compute the z-score of the test statistic.

        Args:
            token_ids (torch.Tensor): Sequence of token indices.
            sample_size (int): Number of samples to estimate the mean and standard deviation.

        Returns:
            float: z-score of the test statistic.
        """
        psi0 = self.test_statistic(token_ids)
        sample_statistics = []

        # Reduce the vocab size by discretizing the token ids
        token_ids_discretized = torch.unique(token_ids, return_inverse=True)[1]
        vocab_size_discretized = max(token_ids_discretized) + 1

        # Generate sample statistics
        for _ in range(n_samples):
            wat = ExpWatermarker(
                lam=self.lam,
                vocab_size=vocab_size_discretized,
                block_size=self.block_size,
                gamma=self.gamma,
                shift=self.shift,
                generator=self.gen,
                device=self.device,
            )
            psi = wat.test_statistic(token_ids_discretized)
            sample_statistics.append(psi)

        # Compute mean and standard deviation of the sample statistics
        mu = np.mean(sample_statistics)
        sigma = np.std(sample_statistics, ddof=1)

        # Calculate z-score
        if sigma == 0:
            raise ValueError(
                "Standard deviation of sample statistics is zero, cannot compute z-score."
            )

        z_score = (psi0 - mu) / sigma

        return z_score.item()

    def p_value(self, token_ids: torch.Tensor, n_samples: int = 1000, upperbound=False) -> float:
        """
        Calculate the p-value of the test statistic.

        Args:
            token_ids (torch.Tensor): Sequence of token indices.
            sample_size (int): Number of samples to estimate the mean and standard deviation.

        Returns:
            float: p-value of the test statistic.
        """
        if upperbound:
            z_score = self.z_score(token_ids, n_samples=n_samples)
            if z_score > 0:
                return 1
            elif abs(z_score) >= math.sqrt(5 / 3):
                return 4 / (9 * (z_score**2 + 1))
            else:
                return 4 / (3 * (z_score**2 + 1)) - 1 / 3
        else:
            psi0 = self.test_statistic(token_ids)
            sample_statistics = []

            # Reduce the vocab size by discretizing the token ids
            token_ids_discretized = torch.unique(token_ids, return_inverse=True)[1]
            vocab_size_discretized = max(token_ids_discretized) + 1

            # Generate sample statistics
            for _ in range(n_samples):
                wat = ExpWatermarker(
                    lam=self.lam,
                    vocab_size=vocab_size_discretized,
                    block_size=self.block_size,
                    gamma=self.gamma,
                    shift=self.shift,
                    generator=self.gen,
                    device=self.device,
                )
                psi = wat.test_statistic(token_ids_discretized)
                sample_statistics.append(psi)

            # Compute the p-value
            rank = sum(psi < psi0 for psi in sample_statistics)
            p_value = (rank + 1) / (len(sample_statistics) + 1)
            return p_value

    def p_value_unoptimized(
        self, token_ids: torch.Tensor, n_samples: int = 1000, upperbound: bool=False
    ) -> float:
        """
        Calculate the p-value of the test statistic.

        Args:
            token_ids (torch.Tensor): Sequence of token indices.
            sample_size (int): Number of samples to estimate the mean and standard deviation.
            upperbound (bool): Whether to use the upper bound approximation for p-value.

        Returns:
            float: p-value of the test statistic.
        """
        if upperbound:
            z_score = self.z_score(token_ids, n_samples=n_samples)
            if z_score > 0:
                return 1
            elif abs(z_score) >= math.sqrt(5 / 3):
                return 4 / (9 * (z_score**2 + 1))
            else:
                return 4 / (3 * (z_score**2 + 1)) - 1 / 3
        else:
            psi0 = self.test_statistic(token_ids)
            sample_statistics = []

            # Reduce the vocab size by discretizing the token ids
            # token_ids_discretized = torch.unique(token_ids, return_inverse=True)[1]

            # Generate sample statistics
            for _ in range(n_samples):
                wat = ExpWatermarker(
                    lam=self.lam,
                    # vocab_size=vocab_size_discretized,
                    vocab_size=self.vocab_size,
                    block_size=self.block_size,
                    gamma=self.gamma,
                    shift=self.shift,
                    generator=self.gen,
                    device=self.device,
                )
                psi = wat.test_statistic(token_ids)
                sample_statistics.append(psi)

            # Compute the p-value
            rank = sum(psi < psi0 for psi in sample_statistics)
            p_value = (rank + 1) / (len(sample_statistics) + 1)
            return p_value

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
