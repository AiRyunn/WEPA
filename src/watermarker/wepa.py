import math

import numba
import numpy as np
import torch
from transformers import DynamicCache, LogitsProcessor


@numba.njit(fastmath=True, cache=True)
def _distance_edit_jit(
    token_ids: np.ndarray,
    degree: int,
    cost: np.ndarray,
    gamma_d: float,
    gamma_i: float,
) -> float:
    lam, length = len(cost), len(token_ids)
    # Use scrolling array.
    f0 = np.zeros(lam)
    f1 = np.zeros(lam)

    for i in range(1, length + 1):
        for u in range(lam):
            d0 = cost[u, token_ids[i - 1]].item()

            f1[u] = f0[u] + gamma_d  # Deletion
            for pred in range(u - degree, u):
                # We can safely access f0[pred] here.
                f1[u] = min(f1[u], f0[pred] + d0)  # Substitution

        u_star = np.argmin(f1)

        # Insertion may cause a cycle. We need to handle this separately.
        # After the edges across u_min are removed, the remaining graph is acyclic.
        for u in range(u_star + 1, lam):
            for pred in range(u - degree, u):
                f1[u] = min(f1[u], f1[pred] + gamma_i)  # Insertion

        for u in range(u_star):
            for pred in range(u - degree, u):
                f1[u] = min(f1[u], f1[pred] + gamma_i)  # Insertion
        f0, f1 = f1, f0

    return f0.min()


class WepaLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        lam: int,
        vocab_size: int,
        substates: torch.Tensor,
        degree: int = 1,
        bits: int | None = None,
        generator: torch.Generator | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the WEPA logits processor.

        Args:
            lam (int): Number of states in the PA.
            vocab_size (int): Size of the vocabulary.
            substates (torch.Tensor): Substates for the watermark.
            degree (int): Degree of the watermark. Defaults to 1.
            bits (int): Bitwidth. Defaults to None.
            device (torch.device): Device to store the watermark. Defaults to "cpu".
        """
        self.lam = lam
        self.vocab_size = vocab_size
        self.degree = degree
        self.bits = bits
        self.substates = substates
        self.device = device
        if generator is not None:
            self.rng = generator
        else:
            self.rng = torch.Generator(device=device)
        self.state = 0

    def reset_state(self):
        """Reset the internal state of the logits processor."""
        self.state = int(
            torch.randint(0, self.lam, (1,), generator=self.rng, device=self.device).item()
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        probs = torch.softmax(scores, dim=-1)

        if self.bits is None:
            mu = self.substates[self.state]
        else:
            sampled_variable = torch.rand(
                (self.vocab_size,), generator=self.rng, device=self.device
            )
            mu = (self.substates[self.state] + sampled_variable) / 2**self.bits

        # Update state
        step = (
            int(torch.randint(0, self.degree, (1,), generator=self.rng, device=self.device).item())
            + 1
        )
        self.state = (self.state + step) % self.lam

        # Exponential minimum sampling
        token_id = torch.argmin(-(torch.log(mu)) / probs)

        new_scores = torch.full_like(scores, -float('Inf'))
        new_scores[0, token_id] = 0
        return new_scores


class WepaWatermarker:
    def __init__(
        self,
        lam: int,
        vocab_size: int,
        degree: int = 1,
        bits: int | None = None,
        gamma_d: float = 1,
        gamma_i: float = 2,
        generator: torch.Generator | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the WEPA watermarker.

        Args:
            lam (int): Number of states in the PA.
            vocab_size (int): Size of the vocabulary.
            degree (int): Degree of the watermark. Defaults to 1.
            bits (int): Bidwitdh. Defaults to None.
            gamma_d (float): Deletion cost. Defaults to 0.
            gamma_i (float): Insertion cost. Defaults to 2.
            precision (int): Precision of the watermark. Defaults to 32.
            generator (torch.Generator | None): Generator for random number generation. Defaults to None.
            device (torch.device): Device to store the watermark. Defaults to "cpu".
        """
        self.lam = lam
        self.vocab_size = vocab_size
        self.degree = degree
        self.bits = bits
        self.gamma_d = gamma_d
        self.gamma_i = gamma_i
        self.gen = generator if generator is not None else torch.Generator()
        self.device = device

        if gamma_i < 0:
            raise ValueError("Gamma_i must be non-negative")

        if bits is not None and not 1 <= bits <= 32:
            raise ValueError("Invalid bitwidth")

        if bits is None:
            self.substates = torch.rand((lam, vocab_size), generator=generator, device=device)
            self.cost = np.log(1 - self.substates.cpu().numpy())
        else:
            self.substates = torch.randint(
                0,
                2**bits,
                (lam, vocab_size),
                generator=generator,
                device=device,
            )

            self.cost = np.log(1 - self.substates.cpu().numpy() / 2**bits)

        if not 1 <= degree < self.lam:
            raise ValueError("Invalid degree")

        self.logits_processor = WepaLogitsProcessor(
            lam=lam,
            vocab_size=vocab_size,
            substates=self.substates,
            degree=degree,
            bits=bits,
            generator=generator,
            device=device,
        )

    def distance_edit(self, token_ids: torch.Tensor) -> float:
        """
        Compute the distance for a given sequence of tokens y.
        """
        assert token_ids.ndim == 1, "token_ids must be a 1D tensor"
        return _distance_edit_jit(
            token_ids.cpu().numpy(),
            self.degree,
            self.cost,
            self.gamma_d,
            self.gamma_i,
        )

    @torch.no_grad()
    def generate(
        self,
        model: torch.nn.Module,
        inputs: dict,
        max_new_tokens: int,
    ):
        """
        Generate samples using the WEPA watermarking scheme.

        Args:
            model (torch.nn.Module): The model to generate samples from.
            input_ids (torch.Tensor): The input sequence to condition on.
            max_length (int): The maximum length of the generated sequence.
            return_entropy (bool): Whether to return the empirical entropy of the generated sequence.
            force_max_length (bool): Whether to force the generated sequence to be of length `max_length`.

        Returns:
            list[int]: The generated sequence of token indices.
        """
        batch_size = inputs['input_ids'].shape[0]
        assert batch_size == 1, "Batch size > 1 is not supported"

        model.eval()
        self.logits_processor.reset_state()

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
        return self.distance_edit(token_ids)

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
            wat = WepaWatermarker(
                lam=self.lam,
                vocab_size=vocab_size_discretized,
                # vocab_size=self.vocab_size,
                degree=self.degree,
                bits=self.bits,
                gamma_d=self.gamma_d,
                gamma_i=self.gamma_i,
                generator=self.gen,
                device=self.device,
            )
            psi = wat.test_statistic(token_ids_discretized)
            # psi = wat.test_statistic(token_ids)
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

    def p_value(
        self, token_ids: torch.Tensor, n_samples: int = 1000, upperbound: bool = False
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
            token_ids_discretized = torch.unique(token_ids, return_inverse=True)[1]
            vocab_size_discretized = max(token_ids_discretized) + 1

            # Generate sample statistics
            for _ in range(n_samples):
                wat = WepaWatermarker(
                    lam=self.lam,
                    vocab_size=vocab_size_discretized,
                    degree=self.degree,
                    bits=self.bits,
                    gamma_d=self.gamma_d,
                    gamma_i=self.gamma_i,
                    generator=self.gen,
                    device=self.device,
                )
                psi = wat.test_statistic(token_ids_discretized)
                sample_statistics.append(psi)

            # Compute the p-value
            rank = sum(psi <= psi0 for psi in sample_statistics)
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
