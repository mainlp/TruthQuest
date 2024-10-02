"""
Classes and modules to evaluate the model's conclusion
"""

import logging
import re
from typing import Any

from metareasoning.evaluator.evaluator_base import Evaluator

ENTITIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]


class ConclusionEvaluator(Evaluator):
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
    def rule_based_evaluation(
        model_reponse: str, context: dict[str, str], num_characters: int
    ) -> dict[str, bool | None]:
        """
        Extract final conclusion from model's response.

        Args:
            model_reponse (str): The response of the model.
            context (dict[str, str]): The context, i.e. descriptions for truthteller and liars.
            num_characters (int): The number of characters considered in the problem statetement.

        Returns:
            dict[str, bool]: The extract final conclusion. Could be empty if no unique conclusion could be found.
        """
        entities = f": ({context['truth-teller']}|{context['liar']})\n".join(
            ENTITIES[:num_characters]
        )
        pattern = f"^{entities.lower()}: ({context['truth-teller']}|{context['liar']})\s*$"  # noqa: W605
        matches = re.findall(pattern, model_reponse.lower(), re.MULTILINE)

        if not matches:
            return {}

        # If more than one match is found, check if they are all the same
        if len(matches) > 1:
            if len(set(matches)) != 1:
                return {}

        context_mapping = {context["truth-teller"]: True, context["liar"]: False}

        extraction: dict[str, bool | None] = {
            ENTITIES[character_id]: context_mapping[identity]
            for character_id, identity in enumerate(matches[0])
        }
        return extraction

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
        try:
            decoded_input, decoded_output = self.forward(
                encoded_input=encoded_input, inference_kwargs=inference_kwargs
            )
        except Exception as e:
            logging.warn(
                f"Could not run forward conclusion evaluation pass because of the following error:\n{e}"
            )
            print(f"encoded_input: {encoded_input}")
            quit()

        return decoded_input, decoded_output
