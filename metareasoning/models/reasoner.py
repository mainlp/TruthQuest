"""
Reasoning module based on HF models
"""

import logging
import random
from collections import Counter
from typing import Any, Literal

from metareasoning.evaluator.evaluator_conclusion import ConclusionEvaluator
from metareasoning.models.model_wrapper import ModelWrapper

logger = logging.getLogger(__name__)


class Reasoner(ModelWrapper):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        model_init_kwargs: dict[str, Any],
        tokenizer_path: str,
        tokenizer_init_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            model_name,
            model_path,
            model_init_kwargs,
            tokenizer_path,
            tokenizer_init_kwargs,
        )

    def self_consistency(
        self,
        encoded_input: dict[str, Any],
        inference_kwargs: dict[str, Any],
        context: dict[str, str],
        num_chars: int,
        num_reasoning_paths: int = 1,
    ) -> tuple[list[str], list[str]]:
        """
        Implements self-consistency as defined in <https://arxiv.org/abs/2203.11171>.
        In particular, samples over multiple (K) forward passes and applies majority vote on the final answer.

        Args:
            encoded_input (dict[str, Any]): The encoded input for the inference.
            inference_kwargs (dict[str, Any]): Additional keyword arguments for inference.
            num_reasoning_paths (Optional[int]): Number of reasoning paths to consider.

        Returns:
            tuple[list[str], list[str]]: A tuple containing the decoded input and the decoded output.
        """

        decoded_outputs: list[list[str]] = []
        parsed_outputs: list[list[dict[str, bool]]] = []

        # Perform multiple forward passes
        for _ in range(num_reasoning_paths):
            decoded_input, decoded_output = self.forward(
                encoded_input=encoded_input, inference_kwargs=inference_kwargs
            )
            decoded_outputs.append(decoded_output)

            parsed_output = list(
                map(
                    lambda x: ConclusionEvaluator.rule_based_evaluation(
                        x, context=context, num_characters=num_chars
                    ),
                    decoded_output,
                )
            )
            parsed_outputs.append(parsed_output)

        num_samples = len(decoded_outputs[0])
        final_decoded_output: list[str] = []

        # Find the majority vote
        for n in range(num_samples):
            sample_outputs = [
                parsed_outputs[i][n]
                for i in range(num_reasoning_paths)
                if parsed_outputs[i][n]
            ]
            if all(not d for d in sample_outputs):
                final_decoded_output.append(random.choice(decoded_outputs)[n])

            else:
                most_common_tuple, _ = Counter(
                    frozenset(d.items()) for d in sample_outputs
                ).most_common(1)[0]
                most_common = dict(most_common_tuple)
                index_of_most_common = sample_outputs.index(most_common)
                final_decoded_output.append(decoded_outputs[index_of_most_common][n])

        return decoded_input, final_decoded_output

    def inference(
        self,
        encoded_input: dict[str, Any],
        inference_kwargs: dict[str, Any],
        reasoning_strategy: Literal[
            "zero_shot",
            "four_shot",
            "eight_shot",
            "zero_cot",
            "four_cot",
            "eight_cot",
            "cot_sc",
        ] = "zero_cot",
        context: dict[str, str] = {},
        num_chars: int = 0,
        num_reasoning_paths: int = 10,
    ) -> tuple[Any, Any]:
        """
        Perform inference using the specified reasoning strategy.

        Args:
            encoded_input (dict[str, Any]): The encoded input for the inference.
            inference_kwargs (dict[str, Any]): Additional keyword arguments for inference.
            reasoning_strategy (Literal["naive", "zero_cot", "cot", "cot_sc"], optional): The reasoning strategy to use. Defaults to "zero_cot".
            reasoning_strategy (Literal["zero_shot", "four_shot", "eight_shot", "zero_cot", "eight_cot", "cot_sc"], optional): The reasoning strategy to use. Defaults to "zero_cot".
            context (dict[str, str]): The context of the puzzle.
            num_chars (int): The number of characters considered in the puzzle.
            num_reasoning_paths (int): The number of reasoning paths to sample in self-consistency.

        Returns:
            Tuple[Any, Any]: A tuple containing the decoded input and the decoded output.
        """
        if reasoning_strategy in [
            "zero_shot",
            "four_shot",
            "eight_shot",
            "zero_cot",
            "four_cot",
            "eight_cot",
        ]:
            decoded_input, decoded_output = self.forward(
                encoded_input=encoded_input, inference_kwargs=inference_kwargs
            )
        elif reasoning_strategy == "cot_sc":
            decoded_input, decoded_output = self.self_consistency(
                encoded_input,
                inference_kwargs,
                context=context,
                num_chars=num_chars,
                num_reasoning_paths=num_reasoning_paths,
            )
        else:
            error_message = f"Invalid reasoning strategy: {reasoning_strategy}"
            logging.error(error_message)
            raise ValueError(error_message)

        return decoded_input, decoded_output
