import abc

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


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

    def tpr_at_fpr(
        self,
        neg_input_ids_list: list[torch.Tensor],
        pos_input_ids_list: list[torch.Tensor],
        fpr_target: float = 0.01,
    ) -> float:
        """
        Compute the true positive rate (TPR) at a given false positive rate (FPR) target.
        """
        all_statistics, labels = self.scores(neg_input_ids_list, pos_input_ids_list)
        # Calculate the threshold for the desired FPR
        neg_statistics = all_statistics[labels == 1]
        threshold = np.percentile(neg_statistics, 100 * (1 - fpr_target))
        # Calculate TPR at this threshold
        pos_statistics = all_statistics[labels == 0]
        tpr = np.mean(pos_statistics >= threshold)
        return float(tpr)

    def roc_auc(
        self,
        neg_input_ids_list: list[torch.Tensor],
        pos_input_ids_list: list[torch.Tensor],
    ) -> float:
        """
        Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC).
        """
        all_statistics, labels = self.scores(neg_input_ids_list, pos_input_ids_list)
        auc = roc_auc_score(labels, all_statistics)
        return float(auc)
