import numpy as np
import torch


class TopPSampler:
    """
    Class for top-p (nucleus) sampling and text generation.
    """

    def __init__(self):
        """
        Initialize the TopPSampler.
        """
        pass

    @staticmethod
    def _top_p_sampling(logits: torch.Tensor, p: float) -> int:
        """
        Perform nucleus (top-p) sampling from logits.

        Args:
            logits (torch.Tensor): Logits returned by the model, shape (vocab_size,).
            p (float): Cumulative probability threshold.

        Returns:
            int: Selected token ID.
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Sort probabilities and their corresponding token IDs
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Calculate cumulative probabilities
        cumulative_probs = np.cumsum(sorted_probs)

        # Find the smallest set of tokens with cumulative probability >= p
        cutoff_index = np.where(cumulative_probs >= p)[0][0]
        top_indices = sorted_indices[: cutoff_index + 1]
        top_probs = sorted_probs[: cutoff_index + 1]

        # Normalize the top probabilities
        top_probs /= top_probs.sum()

        # Sample from the top probabilities
        selected_index = np.random.choice(top_indices, p=top_probs)
        return selected_index

    def generate(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        max_length: int,
        p: float = 0.9,
    ) -> list[int]:
        """
        Generate text using top-p sampling.

        Args:
            input_ids (torch.Tensor): Input token IDs, shape (1, seq_len).
            max_length (int): Maximum length of the generated sequence.
            p (float): Nucleus sampling probability threshold.

        Returns:
            output_ids (torch.Tensor): Output token IDs, shape (seq_len,).
        """

        # Generate tokens
        output_ids = input_ids.tolist()[0]
        model.eval()

        with torch.no_grad():
            for _ in range(max_length):
                # Get the model's logits for the next token
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :].squeeze(
                    0
                )  # Logits for the last token

                # Apply top-p sampling
                next_token_id = self._top_p_sampling(next_token_logits, p)

                # Add the sampled token to the sequence
                output_ids.append(next_token_id)
                input_ids = torch.tensor([output_ids]).to(input_ids.device)

                # Stop if the end-of-sequence token is generated
                if next_token_id == model.config.eos_token_id:
                    break

        return output_ids
