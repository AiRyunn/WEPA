import abc

import numpy as np
import torch


class Watermarker(abc.ABC):
    """
    Abstract base class for watermarking methods.
    Concrete subclasses must implement __init__ and may override other methods.
    """

    # --- Abstract Methods ---
    @abc.abstractmethod
    def test_statistic(self, token_ids: torch.Tensor) -> float:
        """Compute the test statistic for the given sequence."""
        pass

    def scores(
        self,
        neg_input_ids_list: list[torch.Tensor],
        pos_input_ids_list: list[torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray]:
        neg_statistics = [self.test_statistic(x) for x in neg_input_ids_list]
        pos_statistics = [self.test_statistic(x) for x in pos_input_ids_list]

        neg_statistics = np.array(neg_statistics)
        pos_statistics = np.array(pos_statistics)

        all_statistics = np.concatenate([neg_statistics, pos_statistics])
        labels = np.concatenate([np.ones(len(neg_statistics)), np.zeros(len(pos_statistics))])
        return all_statistics, labels
