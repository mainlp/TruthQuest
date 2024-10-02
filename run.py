import argparse
import os
from typing import Any, Literal

from tqdm import tqdm

from metareasoning.models.model_args import ModelArgs, PromptArgs
from metareasoning.models.reasoner import Reasoner
from metareasoning.prompts.prompt_manager import PromptManager
from metareasoning.utils.utils import (
    inference_pipeline,
    load_args,
    parse_context,
    prepare_dataset_from_disk,
    read_yaml_file,
    set_seed,
    setup_logging,
    write_dataset_to_jsonl,
)

DATA_DIR = os.path.join("data")
PROMPT_PATH = os.path.join("metareasoning", "prompts")
DATA_ARG_PATH = os.path.join("metareasoning", "dataprep", "data_config")
MODEL_ARG_PATH = os.path.join("metareasoning", "models", "model_config")
OUTPUT_DIR = os.path.join("experimental_results")
SAVE_PATH = os.path.join("hf-models")


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Fetch CLI arguments
    parser = argparse.ArgumentParser(
        "Evaluating the Suppositional Reasoning Ability of Large Language Models"
    )

    # General configs
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device"
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument(
        "--answer-only", action="store_true", help="Whether to record only the answer"
    )
    parser.add_argument(
        "--save-text",
        action="store_true",
        help="Whether to record each answer in a separate text file",
    )

    # Configs about experiment
    parser.add_argument(
        "--num-samples", type=int, default=5, help="Number of experiment iterations"
    )
    parser.add_argument(
        "--context",
        "-t",
        type=str,
        default="jabbas",
        choices=["knights", "jabbas", "neutral"],
        help="Context of the puzzle, either knights & knaves, jabbas & tettes, or truth-tellers and liars.",
    )

    # Configs about model
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Large Language Model to use",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="zero_shot",
        choices=[
            "zero_shot",
            "four_shot",
            "eight_shot",
            "zero_cot",
            "four_cot",
            "eight_cot",
            "cot_sc",
        ],
        help="LLM reasoning strategy",
    )

    return parser.parse_args()


def load_config(model_name: str) -> tuple[ModelArgs, dict[str, Any], PromptArgs]:
    """
    Loads the configuration for the model, tokenizer and prompts based on the provided model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        tuple[ModelArgs, dict[str, Any], PromptArgs]: A tuple containing the model arguments, tokenizer arguments, and prompt arguments.
    """
    model_config_file = os.path.join(
        MODEL_ARG_PATH, f"{model_name.replace('/', '_')}.yaml"
    )
    return load_args(model_config_file)


def initialize_reasoner(args: argparse.Namespace) -> Reasoner:
    """
    Initializes the model wrapper based on the provided arguments.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        Reasoner: The initialized model wrapper.
    """
    # model & tokenizer
    model_args, tokenizer_kwargs, _ = load_config(args.model)

    model_path = os.path.join(SAVE_PATH, "model", args.model)
    tokenizer_path = os.path.join(SAVE_PATH, "tokenizer", args.model)

    reasoner = Reasoner(
        model_name=args.model,
        model_path=model_path,
        model_init_kwargs=model_args.init_kwargs,
        tokenizer_path=tokenizer_path,
        tokenizer_init_kwargs=tokenizer_kwargs,
    )

    return reasoner


def load_prompts(
    prompt_manager: PromptManager,
    reasoning_strategy: Literal[
        "zero_shot",
        "four_shot",
        "eight_shot",
        "zero_cot",
        "four_cot",
        "eight_cot",
        "cot_sc",
    ],
    cautious_mode: bool = False,
    prompt_subdir: str = "",
) -> tuple[str, str, list[str]]:
    """
    Load prompts for a given reasoning strategy and configuration.

    Args:
        prompt_manager (PromptManager): An instance of the PromptManager class.
        reasoning_strategy (Literal["zero_shot", "four_shot", "eight_shot", "zero_cot", "four_cot", "eight_cot", "cot_sc"]): The reasoning strategy to use.
        cautious_mode (bool, optional): Whether to enable cautious mode. Defaults to False.
        prompt_subdir (str, optional): The subdirectory to use for prompts. Defaults to "".

    Returns:
        tuple[str, str, list[str]]: A tuple containing the system message, prompt, and potentially a list of few-shot examples.
    """
    task_prompt_dir = os.path.join(PROMPT_PATH, "task_prompts")
    strategy_prompt_dir = os.path.join(PROMPT_PATH, "reasoning_prompts")

    # task prompt files
    sys_message_file = os.path.join(task_prompt_dir, "system_message.txt")
    prompt_file = os.path.join(task_prompt_dir, "prompt.txt")

    # reasoning prompts
    if reasoning_strategy in ["four_shot", "eight_shot", "four_cot", "eight_cot"]:
        few_shot_prompt_file = os.path.join(
            strategy_prompt_dir,
            "few_shot" if reasoning_strategy in ["four_shot", "eight_shot"] else "cot",
            prompt_subdir,
            "few_shot_prompt.txt",
        )
        num_shots = 4 if "four" in reasoning_strategy else 8
    else:
        few_shot_prompt_file = None
        num_shots = 0

    suffix_prompt_file = (
        os.path.join(
            strategy_prompt_dir, reasoning_strategy, f"{reasoning_strategy}_prompt.txt"
        )
        if reasoning_strategy == "zero_cot"
        else None
    )

    cautious_sys_message_file = (
        os.path.join(task_prompt_dir, "cautious_system_instruction.txt")
        if cautious_mode
        else None
    )

    # get input prompts
    sys_message, prompt, few_shot_examples = prompt_manager.get_input_prompts(
        sys_message_file_path=sys_message_file,
        prompt_file_path=prompt_file,
        suffix_prompt_file_path=suffix_prompt_file,
        few_shot_prompt_file_path=few_shot_prompt_file,
        cautious_sys_message_file_path=cautious_sys_message_file,
        num_shots=num_shots,
    )

    return sys_message, prompt, few_shot_examples


def run_inference(
    args: argparse.Namespace,
    dataset: Any,
    reasoner: Reasoner,
    output_path: str,
    prompt_subdir: str,
) -> None:
    """
    Runs the inference pipeline on the dataset using the reasoner.

    Args:
        args (argparse.Namespace): Command line arguments.
        dataset (Any): The prepared dataset.
        reasoner (Reasoner): The initialized model wrapper.
    """
    model_args, tokenizer_kwargs, prompt_args = load_config(args.model)

    if "Meta-LLama-3" in args.model:
        terminators = [
            reasoner.tokenizer.eos_token_id,
            reasoner.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
    else:
        terminators = reasoner.tokenizer.eos_token_id

    model_args.inference_kwargs.update(
        {
            "eos_token_id": terminators,
            "pad_token_id": reasoner.tokenizer.pad_token_id,
        }
    )

    # get task prompts
    prompt_manager = PromptManager()
    sys_message, prompt, few_shot_examples = load_prompts(
        prompt_manager=prompt_manager,
        reasoning_strategy=args.strategy,
        cautious_mode=prompt_args.cautious_mode,
        prompt_subdir=prompt_subdir,
    )

    # convert prompts and encode
    context = parse_context(args.context)
    num_chars = dataset["metadata"][0]["num_characters"]
    output_key = "task_prompt"

    substitution_dict: dict[str, str] = {
        "<num-characters>": str(num_chars),
        "<statements>": "<statements>",
        "truth-teller": context["truth-teller"],
        "liar": context["liar"],
    }

    dataset = dataset.map(
        prompt_manager.create_chat_prompt,
        fn_kwargs={
            "input_key": "problem",
            "output_key": output_key,
            "user_prompt": prompt,
            "user_prompt_special_token": "<statements>",
            "few_shot_examples": few_shot_examples,
            "substitution_dict": substitution_dict,
            "allow_system_message": prompt_args.system_message,
            "system_message": sys_message,
            "tokenizer": reasoner.tokenizer,
        },
        batched=True,
        batch_size=len(dataset),
        load_from_cache_file=False,
    )
    encoded_input = reasoner.tokenizer(
        dataset[output_key], padding=True, return_tensors="pt"
    ).to(args.device)

    # inference
    for run_nr in tqdm(range(args.num_samples), desc="Processing Samples"):
        generated_output = inference_pipeline(
            encoded_input_dict=encoded_input,
            inference_function=reasoner.inference,
            function_kwargs={
                "inference_kwargs": model_args.inference_kwargs,
                "reasoning_strategy": args.strategy,
                "context": context,
                "num_chars": num_chars,
            },
            batch_size=args.batch_size,
        )

        # add info to dataset
        additional_metadata = {
            "model": args.model,
            "strategy": args.strategy,
            "context": context,
            "sample_id": run_nr,
            "batch_size": args.batch_size,
            "model_init_kwargs": dict(
                (k, v) for k, v in model_args.init_kwargs.items() if k != "torch_dtype"
            ),
            "model_inference_kwargs": model_args.inference_kwargs,
            "tokenizer_kwargs": tokenizer_kwargs,
        }

        dataset = dataset.add_column(
            "model_answer", generated_output["generated_output"]
        )
        dataset = dataset.map(
            lambda instance: {
                **instance,
                "metadata": {**instance["metadata"], **additional_metadata},
            }
        )

        # write results to files
        model_path_name = (
            f"{args.model}_cautious" if prompt_args.cautious_mode else args.model
        )
        output_path = os.path.join(
            output_path,
            model_path_name,
            args.strategy,
        )
        write_dataset_to_jsonl(
            dataset=dataset,
            file_path=os.path.join(output_path, f"model_answers_{run_nr}.jsonl"),
        )

        if args.save_text:
            reasoner.save_results_to_txt(
                folder_path=output_path,
                decoded_output=generated_output["generated_output"],
                decoded_input=generated_output["input"],
            )


def main() -> None:
    """
    Main function to orchestrate the execution flow.
    """
    args = parse_arguments()

    setup_logging(args.verbose)
    set_seed(args.seed)

    # model
    reasoner = initialize_reasoner(args)

    # dataset configs
    dataset_configs = read_yaml_file(os.path.join(DATA_ARG_PATH, "dataset_params.yaml"))
    statement_types = dataset_configs["statement_types"]
    num_characters = dataset_configs["characters"]

    # run inference on dataset
    for statements in tqdm(
        statement_types, desc="Processing statement types", leave=False
    ):
        for characters in num_characters:
            data_config_path = os.path.join(
                f"statements_{''.join(str(s) for s in statements)}",
                f"characters_{characters}",
            )
            dataset_path = os.path.join(
                DATA_DIR,
                data_config_path,
                "puzzles.jsonl",
            )
            dataset = prepare_dataset_from_disk(dataset_path)

            # forward
            output_path = os.path.join(
                OUTPUT_DIR,
                "model_answers",
                args.context,
                data_config_path,
            )
            run_inference(
                args=args,
                dataset=dataset,
                reasoner=reasoner,
                output_path=output_path,
                prompt_subdir=data_config_path,
            )


if __name__ == "__main__":
    main()
