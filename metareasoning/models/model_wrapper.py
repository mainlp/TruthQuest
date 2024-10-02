"""
Model wrapper around HF models
"""

import logging
import os
from typing import Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from metareasoning.utils.utils import save_dict_to_json, write_to_text_file

logger = logging.getLogger(__name__)


class ModelWrapper:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        model_init_kwargs: dict[str, Any],
        tokenizer_path: str,
        tokenizer_init_kwargs: dict[str, Any],
    ) -> None:
        self.model = self._init_model(model_name, model_path, model_init_kwargs)
        self.tokenizer = self._init_tokenizer(
            model_name, tokenizer_path, tokenizer_init_kwargs
        )

    def _init_model(
        self, model_name: str, model_path: str, model_init_kwargs: dict
    ) -> PreTrainedModel:
        """
        Initialize the model with the specified name, path, and initialization arguments.

        :param model_name: The name of the model to be initialized.
        :param model_path: The path to the model.
        :param model_init_kwargs: Additional keyword arguments for model initialization.
        :return: An instance of PreTrainedModel.
        """
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=model_path, device_map="auto", **model_init_kwargs
            )
            logger.info(f"Model {model_name} initialized successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {str(e)}")
            raise

    def _init_tokenizer(
        self, model_name: str, tokenizer_path: str, tokenizer_init_kwargs: dict
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """
        Initialize the tokenizer with the specified model name, tokenizer path, and initialization arguments.

        Args:
            model_name (str): The name of the model to be used for tokenization.
            tokenizer_path (str): The path to the tokenizer.
            tokenizer_init_kwargs (dict): Additional initialization keyword arguments for the tokenizer.

        Returns:
            PreTrainedTokenizer | PreTrainedTokenizerFast: The initialized tokenizer.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=tokenizer_path, **tokenizer_init_kwargs
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Tokenizer for {model_name} initialized successfully.")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer for {model_name}: {str(e)}")
            raise

    @staticmethod
    def format_output(decoded_answer: list[str]) -> tuple[list[str], list[str]]:
        """
        Method to format the output.

        Args:
            decoded_answer (list[str]): The list of decoded answers to be formatted.

        Returns:
            tuple[list[str], list[str]]: A tuple containing two lists, the formatted decoded input and decoded output.
        """
        instruction_start_tokens: list[str] = ["[INST]", "user\n\n"]
        assistant_start_tokens = [
            "[/INST] ",
            "<|im_start|> assistant\n",
            "<|assistant|>",
            "assistant\n\n",
        ]
        decoded_input = []
        decoded_output = []

        for sample in decoded_answer:
            # get final shot
            for instr_start_token in instruction_start_tokens:
                if instr_start_token in sample:
                    all_shots = sample.split(instr_start_token)
                    few_shots = (
                        f"{instr_start_token}".join(all_shots[:-1]) + instr_start_token
                    )
                    final_shot = all_shots[-1]

                    # get model answer
                    for ass_start_token in assistant_start_tokens:
                        if ass_start_token in final_shot:
                            parts = final_shot.split(ass_start_token)
                            assert (
                                len(parts) == 2
                            ), f"More than one instruction token found!\n{sample}\nWith token: {ass_start_token}"
                            decoded_input.append(few_shots + parts[0] + ass_start_token)
                            decoded_output.append(parts[1])
                            break

        return decoded_input, decoded_output

    def forward(
        self,
        encoded_input: dict[str, Any],
        inference_kwargs: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        """
        Perform forward inference using the encoded input provided and inference arguments.

        Args:
            encoded_input (dict[str, Any]): The encoded input for the inference.
            inference_kwargs (dict[str, Any]): Additional keyword arguments for the inference.

        Returns:
            tuple[list[str], list[str]]: The decoded input and decoded output as lists of strings.
        """
        outputs = self.model.generate(**encoded_input, **inference_kwargs)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_input, decoded_output = self.format_output(outputs)

        return decoded_input, decoded_output

    @staticmethod
    def save_results_to_json(
        result_dict: dict[str, Any], file_path: str, metadata: dict[str, Any] = {}
    ) -> None:
        """
        Save the results to a JSON file.

        Args:
            result_dict (dict[str, list[str]]): A dictionary containing the results to be saved.
            file_path (str): The file path where the results will be saved.
            metadata (dict[str, Any], optional): Additional metadata to be included in the saved results. Defaults to {}.

        Returns:
            None
        """
        metadata_dict: dict[str, Any] = {"metadata": metadata}
        result_dict.update(metadata_dict)
        save_dict_to_json(result_dict, file_path=file_path)

    @staticmethod
    def save_results_to_txt(
        folder_path: str,
        decoded_output: list[str],
        decoded_input: list[str] | None = None,
    ) -> None:
        """
        Save the results to a text file.

        Args:
            folder_path (str): The path to the folder where the text files will be saved.
            decoded_output (list[str]): The list of decoded output strings.
            decoded_input (list[str] | None, optional): The list of decoded input strings.
                Defaults to None.

        Returns:
            None: This function does not return anything.
        """
        for task_nr, out_text in enumerate(decoded_output):
            in_text = "" if decoded_input is None else decoded_input[task_nr]
            write_to_text_file(
                content=f"{in_text}{out_text}",
                file_path=os.path.join(folder_path, f"task{task_nr}.txt"),
            )
