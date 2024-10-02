"""
Classes for model answer evaluations
"""

from typing import Any

from metareasoning.models.model_wrapper import ModelWrapper


class Evaluator(ModelWrapper):
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

    def inference(
        self, encoded_input: dict[str, Any], inference_kwargs: dict[str, Any]
    ) -> tuple[Any, Any]:
        """
        Perform inference.

        Args:
            encoded_input (dict[str, Any]): The encoded input for the inference.
            inference_kwargs (dict[str, Any]): Additional keyword arguments for inference.

        Returns:
            Tuple[Any, Any]: A tuple containing the decoded input and the decoded output.
        """
        decoded_input, decoded_output = self.forward(
            encoded_input=encoded_input, inference_kwargs=inference_kwargs
        )

        return decoded_input, decoded_output
