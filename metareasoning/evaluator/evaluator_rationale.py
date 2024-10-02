"""
Classes and modules to evaluate the model's rationale
"""

import re
from typing import Any

from metareasoning.evaluator.evaluator_base import Evaluator


class RationaleEvaluator(Evaluator):
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

    @staticmethod
    def parse_error_labels(evaluator_response: str, error_type: str) -> bool:
        """
        Extract final conclusion from model's response.

        Args:
            model_reponse (str): The response of the model.
            context (dict[str, str]): The context, i.e. descriptions for truthteller and liars.
            num_characters (int): The number of characters considered in the problem statetement.

        Returns:
            dict[str, bool]: The extract final conclusion. Could be empty if no unique conclusion could be found.
        """
        section_pattern = rf"### {error_type}(.*?)(?=###|$)"
        section_match = re.search(
            section_pattern, evaluator_response, re.DOTALL | re.IGNORECASE
        )

        if section_match:
            section = section_match.group(1)
            label_pattern = r"- Label:\s*(yes|no)"
            label_match = re.search(label_pattern, section)

            if label_match:
                if label_match.group(1) == "yes":
                    return True
                elif label_match.group(1) == "no":
                    return False
                else:
                    raise ValueError(f"Invalid label: {label_match}")
            else:
                raise ValueError(
                    f"Label is not yes/no for error type: {error_type} in text:\n{evaluator_response}"
                )
        else:
            raise ValueError(
                f"Error classification not found for error type: {error_type} in text:\n{evaluator_response}"
            )

    @staticmethod
    def parse_human_error_labels(human_annotation: str, error_type: str) -> bool:
        """
        Extract final conclusion from model's response.

        Args:
            model_reponse (str): The response of the model.
            context (dict[str, str]): The context, i.e. descriptions for truthteller and liars.
            num_characters (int): The number of characters considered in the problem statetement.

        Returns:
            dict[str, bool]: The extract final conclusion. Could be empty if no unique conclusion could be found.
        """
        section_pattern = rf"{error_type}:\s*(yes|no)\s*"
        section_match = re.search(section_pattern, human_annotation, re.DOTALL)

        if section_match:
            label = section_match.group(1)
            if label == "yes":
                return True
            elif label == "no":
                return False
            else:
                raise ValueError(f"Invalid label: {label}")
        else:
            raise ValueError(
                f"Error classification not found for error type: {error_type} in text:\n{human_annotation}"
            )
