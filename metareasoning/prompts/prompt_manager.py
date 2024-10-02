"""
Interface to handle different LLMs
"""

import logging
from typing import Any, Literal

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from metareasoning.utils.utils import read_text_file


class PromptManager:
    def __init__(self):
        pass

    @staticmethod
    def load_few_shot_prompt(
        few_shot_prompt_file_path: str, num_shots: int
    ) -> list[str]:
        """
        Loads the few-shot prompt from the specified file path and returns a string.
        Expects few-shot-examples to be separated by "---".

        Args:
            few_shot_prompt_file_path (str): The file path of the few-shot prompt.
            num_shots (int): The number of shots to retrieve from the few-shot prompt.

        Returns:
            list[str]: The few-shot examples in a list. Note that the each few-shot example is composed of <user> message and <assistant> response.
        """
        few_shot_prompt = read_text_file(few_shot_prompt_file_path)

        if "---" not in few_shot_prompt:
            logging.warn(
                f"Few-shot-examples are not separated by '---':\n{few_shot_prompt}"
            )

        return few_shot_prompt.split("---")[:num_shots]

    @staticmethod
    def load_prompts(
        sys_message_file_path: str,
        prompt_file_path: str,
        prefix_prompt_file_path: str | None,
        suffix_prompt_file_path: str | None,
        few_shot_prompt_file_path: str | None = None,
        num_shots: int = 0,
    ) -> tuple[str, str | None, str, str | None, list[str]]:
        """Loads the system message, prompt, prompt prefix, and prompt suffix from the specified file paths.

        Args:
            sys_message_file_path (str): The file path of the system message file.
            prompt_file_path (str): The file path of the prompt file.
            prefix_prompt_file_path (str | None): The file path of the prompt prefix file (optional).
            suffix_prompt_file_path (str | None): The file path of the prompt suffix file (optional).
            few_shot_prompt_file_path (str | None): The file path for few-shot examples (optional).

        Returns:
            tuple[str, str | None, str, str | None, list[str]]: A tuple containing the system message, prompt prefix, prompt, prompt suffix, and few-shot examples.
        """
        sys_message = read_text_file(sys_message_file_path)
        prompt = read_text_file(prompt_file_path)
        prompt_prefix = (
            read_text_file(prefix_prompt_file_path) if prefix_prompt_file_path else None
        )
        prompt_suffix = (
            read_text_file(suffix_prompt_file_path) if suffix_prompt_file_path else None
        )
        few_shot_examples = (
            PromptManager.load_few_shot_prompt(few_shot_prompt_file_path, num_shots)
            if few_shot_prompt_file_path
            else []
        )

        return sys_message, prompt_prefix, prompt, prompt_suffix, few_shot_examples

    @staticmethod
    def assemble_prompt(
        prompt: str,
        prompt_prefix: str | None = None,
        prompt_suffix: str | None = None,
        prompt_mapping: dict[str, str] = {},
    ) -> str:
        """
        Assembles the prompt by adding a prefix and suffix to the given prompt.

        Args:
            prompt (str): The original prompt.
            prompt_prefix (str | None, optional): The prefix to add to the prompt. Defaults to None.
            prompt_suffix (str | None, optional): The suffix to add to the prompt. Defaults to None.

        Returns:
            str: The assembled prompt.
        """
        if prompt_prefix is not None:
            prompt = f"{prompt_prefix}\n\n{prompt}"

        # add specifics for prompt type
        if prompt_suffix is not None:
            prompt += f"\n{prompt_suffix}"

        # replace any special tokens with prompt mapping
        for special_token, replacement in prompt_mapping.items():
            prompt = prompt.replace(special_token, replacement)

        return prompt

    def get_input_prompts(
        self,
        sys_message_file_path: str,
        prompt_file_path: str,
        prefix_prompt_file_path: str | None = None,
        suffix_prompt_file_path: str | None = None,
        few_shot_prompt_file_path: str | None = None,
        cautious_sys_message_file_path: str | None = None,
        num_shots: int = 0,
        prompt_mapping: dict[str, Any] = {},
    ) -> tuple[str, str, list[str]]:
        """
        Get input prompts from the specified file paths and return the system message and prompt.

        Args:
            sys_message_file_path (str): The file path for system message.
            prompt_file_path (str): The file path for prompt.
            prefix_prompt_file_path (str | None): The file path for prompt prefix. Defaults to None.
            suffix_prompt_file_path (str | None): The file path for prompt suffix. Defaults to None.
            few_shot_prompt_file_path (str | None): The file path for few-shot examples. Defaults to None.
            cautious_sys_message_file_path (str | None): The file path for cautious system message. Defaults to None.
            num_shots (int, optional): The number of shots to consider. Defaults to 0.
            prompt_mapping (dict[str, Any]): The mapping of prompts to their corresponding values. Defaults to an empty dictionary.

        Returns:
            tuple[str, str, list[str]]: A tuple containing the system message, the prompt, and potentially a list of few-shot examples.
        """
        sys_message, prompt_prefix, prompt, prompt_suffix, few_shot_examples = (
            self.load_prompts(
                sys_message_file_path=sys_message_file_path,
                prompt_file_path=prompt_file_path,
                prefix_prompt_file_path=prefix_prompt_file_path,
                suffix_prompt_file_path=suffix_prompt_file_path,
                few_shot_prompt_file_path=few_shot_prompt_file_path,
                num_shots=num_shots,
            )
        )

        if cautious_sys_message_file_path:
            prompt_prefix = sys_message
            sys_message = read_text_file(cautious_sys_message_file_path)

        prompt = self.assemble_prompt(
            prompt=prompt,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
            prompt_mapping=prompt_mapping,
        )

        return sys_message, prompt, few_shot_examples

    @staticmethod
    def chat_format_few_shot_examples(
        few_shots: list[str],
        user_start_token: str,
        user_end_token: str,
        assistant_start_token: str,
        assistant_end_token: str,
    ) -> list[dict[str, str]]:
        """
        Formats few-shot examples for a chat interface based on start and end tokens.

        Args:
            few_shots (list[str]): List of few-shot examples to format.
            user_start_token (str): Start token for user content.
            user_end_token (str): End token for user content.
            assistant_start_token (str): Start token for assistant content.
            assistant_end_token (str): End token for assistant content.

        Returns:
            list[dict[str, str]]: Formatted few-shot examples with user and assistant roles.
        """
        formatted_few_shots: list[dict[str, str]] = []

        for few_shot_example in few_shots:
            user_dict = {
                "role": "user",
                "content": few_shot_example.split(user_start_token)[-1].split(
                    user_end_token
                )[0],
            }
            assistant_dict = {
                "role": "assistant",
                "content": few_shot_example.split(assistant_start_token)[-1].split(
                    assistant_end_token
                )[0],
            }
            formatted_few_shots += [user_dict, assistant_dict]

        return formatted_few_shots

    def contextualize_few_shots(
        self,
        few_shots: list[str],
        user_prompt: str,
        substitution_dict: dict[str, str],
        user_prompt_special_token: str,
        user_start_token: str = "<user>\n",
        user_end_token: str = "\n</user>",
        assistant_start_token: str = "<assistant>\n",
        assistant_end_token: str = "\n</assistant>",
    ) -> list[dict[str, str]]:
        """
        Contextualizes a list of few-shot examples by formatting them into a chat format and substituting tokens in the content.

        Args:
            few_shots (list[str]): A list of few-shot examples.
            user_prompt (str): The user prompt to substitute tokens with.
            substitution_dict (dict[str, str]): A dictionary mapping tokens to their substitution values.
            user_prompt_special_token (str): The special token to be replaced in the user's prompt.
            user_start_token (str, optional): The starting token for user messages in the chat format. Defaults to "<user>\n".
            user_end_token (str, optional): The ending token for user messages in the chat format. Defaults to "\n</user>".
            assistant_start_token (str, optional): The starting token for assistant messages in the chat format. Defaults to "<assistant>\n".
            assistant_end_token (str, optional): The ending token for assistant messages in the chat format. Defaults to "\n</assistant>".

        Returns:
            list[dict[str, str]]: A list of formatted few-shot examples with substituted tokens in the content.
        """
        formatted_few_shots: list[dict[str, str]] = self.chat_format_few_shot_examples(
            few_shots=few_shots,
            user_start_token=user_start_token,
            user_end_token=user_end_token,
            assistant_start_token=assistant_start_token,
            assistant_end_token=assistant_end_token,
        )

        for few_shot_example in formatted_few_shots:
            if few_shot_example["role"] == "user":
                substitution_dict[user_prompt_special_token] = few_shot_example.get(
                    "content", ""
                )
                few_shot_example["content"] = self.substitute_tokens(
                    user_prompt, substitution_dict
                )
            else:
                few_shot_response = few_shot_example.get("content", "")
                few_shot_example["content"] = self.substitute_tokens(
                    few_shot_response, substitution_dict
                )

        return formatted_few_shots

    @staticmethod
    def substitute_tokens(text: str, substitution_dict: dict[str, str]) -> str:
        """
        Substitute tokens in the input text based on the given substitution dictionary.

        Args:
            text (str): The input text to perform token substitution on.
            substitution_dict (dict[str, str]): A dictionary mapping special tokens to their corresponding replacements.

        Returns:
            str: The text with tokens replaced according to the substitution dictionary.
        """
        for special_token, sub_txt in substitution_dict.items():
            text = text.replace(special_token, sub_txt)

        return text

    def create_chat_prompt(
        self,
        instances: dict[str, Any],
        input_key: str,
        output_key: str,
        user_prompt: str,
        user_prompt_special_token: str,
        few_shot_examples: list[str],
        allow_system_message: bool,
        system_message: str,
        substitution_dict: dict[str, str],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        tokenize: bool = False,
        return_tensors: Literal["pt"] = "pt",
    ) -> dict[str, list]:
        """
        Creates a prompt for a chat-based system by contextualizing few shot examples and the user's prompt.

        Args:
            instances (dict[str, Any]): A dictionary containing input instances.
            input_key (str): The key to access the input instances.
            output_key (str): The key to store the final prompts.
            user_prompt (str): The user's prompt.
            user_prompt_special_token (str): The special token to be replaced in the user's prompt.
            few_shot_examples (list[str]): A list of few shot examples.
            allow_system_message (bool): Whether to allow a system message.
            system_message (str): The system message.
            substitution_dict (dict[str, str]): A dictionary mapping special tokens to their corresponding replacements.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer to use.
            tokenize (bool, optional): Whether to tokenize the prompt. Defaults to False.
            return_tensors (Literal["pt"], optional): The format of the returned tensors. Defaults to "pt".

        Returns:
            dict[str, list]: A dictionary containing the final prompts.

        """
        final_prompts: list[str | int] = []

        # contextualize few shots
        formatted_few_shots = self.contextualize_few_shots(
            few_shots=few_shot_examples,
            user_prompt=user_prompt,
            substitution_dict=substitution_dict,
            user_prompt_special_token=user_prompt_special_token,
        )

        # contextualize final user prompt
        for instance in instances[input_key]:
            substitution_dict[user_prompt_special_token] = (
                "\n".join(instance) if isinstance(instance, list) else instance
            )
            contextualize_prompt = self.substitute_tokens(
                user_prompt, substitution_dict
            )
            user_message = [{"role": "user", "content": contextualize_prompt}]
            messages: list[dict[str, str]] = formatted_few_shots + user_message

            # handle system prompt
            if allow_system_message:
                system_chat_message = [{"role": "system", "content": system_message}]
                messages = system_chat_message + messages
            else:
                messages[0]["content"] = f"{system_message}\n\n{messages[0]['content']}"

            chat = tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=True,
                return_tensors=return_tensors,
            )
            final_prompts.append(chat)

        return {output_key: final_prompts}
