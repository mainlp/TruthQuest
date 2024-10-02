"""
Evaluate the model responses
"""

import argparse
import logging
import os
from typing import Any, Literal

from datasets import Dataset
from tqdm import tqdm

from metareasoning.evaluator.evaluator_conclusion import ConclusionEvaluator, Evaluator
from metareasoning.prompts.prompt_manager import PromptManager
from metareasoning.utils.utils import (
    inference_pipeline,
    load_args,
    load_json,
    prepare_dataset_from_disk,
    read_yaml_file,
    set_seed,
    setup_logging,
    write_dataset_to_jsonl,
)

MODEL_ANSWER_DIR = os.path.join("experimental_results", "model_answers")
PROMPT_DIR = os.path.join("metareasoning", "prompts")
DATA_ARG_DIR = os.path.join("metareasoning", "dataprep", "data_config")
EVALUATOR_ARG_DIR = os.path.join("metareasoning", "evaluator", "evaluator_config")
OUTPUT_DIR = os.path.join("experimental_results", "evaluation", "conclusion_analysis")
SAVE_DIR = os.path.join("hf-models")


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

    # Configs about evaluator
    parser.add_argument(
        "--evaluator",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Large Language Model as evaluator to use.",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=1,
        help="Number of few-shot examples for evaluator.",
    )

    # Configs about evaluation
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B-Instruct",
        ],
        help="Large Language Model to evaluate.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["zero_shot"],
        help="LLM reasoning strategy to evaluate.",
    )
    parser.add_argument(
        "--context",
        "-t",
        type=str,
        default="jabbas",
        choices=["knights", "jabbas", "neutral"],
        help="Context of the puzzle, either knights & knaves, jabbas & tettes, or truth-tellers and liars.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1, help="Number of experiment to evaluate."
    )

    return parser.parse_args()


def reorder_list(elements: list, indices: list) -> list:
    """
    Reorders the list of elements according to the provided indices.

    Args:
        elements (list): The original list of elements.
        indices (list): A list of indices indicating the new order.

    Returns:
        list: The reordered list of elements.
    """
    if len(elements) != len(indices):
        raise ValueError(
            "Both the elements and indices lists must have the same length."
        )

    # Initialize a result list with None values
    reordered = [None] * len(elements)

    # Assign each element to its new index position as specified by `indices`
    for original_index, new_index in enumerate(indices):
        reordered[new_index] = elements[original_index]

    return reordered


def load_existing_stats(filepath: str) -> dict[str, dict]:
    """
    Tries to load existing statistics JSON file; returns an empty dictionary if it fails.
    """
    try:
        return load_json(filepath)
    except FileNotFoundError:
        return {}


def initialize_evaluator(args: argparse.Namespace) -> ConclusionEvaluator:
    """
    Initialize and return a ConclusionEvaluator based on the provided arguments.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        ConclusionEvaluator: The initialized evaluator.
    """
    model_config_path = os.path.join(
        EVALUATOR_ARG_DIR, f"{args.evaluator.replace('/', '_')}.yaml"
    )
    evaluator_args, tokenizer_kwargs, _ = load_args(model_config_path)
    model_path = os.path.join(SAVE_DIR, "model", args.evaluator)
    tokenizer_path = os.path.join(SAVE_DIR, "tokenizer", args.evaluator)

    return ConclusionEvaluator(
        model_name=args.evaluator,
        model_path=model_path,
        model_init_kwargs=evaluator_args.init_kwargs,
        tokenizer_path=tokenizer_path,
        tokenizer_init_kwargs=tokenizer_kwargs,
    )


def compute_accuracy(extracted_conclusions: dict[str, Any]) -> float:
    """
    Compute the accuracy of the model's conclusions compared to the ground truth conclusions.

    This function assumes that each entry in the 'model_conclusion' and 'gt_conclusion' lists corresponds to
    the conclusions from a single evaluation instance. The function calculates the proportion of instances
    where the model's conclusions match the ground truth exactly.

    Parameters:
        extracted_conclusions (dict[str, List[Dict[str, bool]]]): A dictionary containing lists of conclusions.
            The keys are 'model_conclusion' and 'gt_conclusion', where each list contains dictionaries
            representing a single conclusion set with characters as keys and a boolean indicating truthfulness.

    Returns:
        float: The accuracy of the model's conclusions as a percentage, represented as a float between 0 and 1.
    """
    model_conclusions = extracted_conclusions["model_conclusion"]
    gt_conclusions = extracted_conclusions["gt_conclusion"]

    if not model_conclusions or not gt_conclusions:
        raise ValueError(
            "Both model and ground truth conclusions must be provided and non-empty."
        )

    if len(model_conclusions) != len(gt_conclusions):
        raise ValueError(
            "The number of model conclusions and ground truth conclusions must match."
        )

    correct_predictions = 0
    total_conclusions = len(gt_conclusions)

    for pred, ground_truth in zip(model_conclusions, gt_conclusions):
        if all(
            ground_truth.get(character) == pred.get(character)
            for character in ground_truth
        ):
            correct_predictions += 1

    return correct_predictions / total_conclusions


def run_evaluator(
    args: argparse.Namespace,
    dataset: Any,
    context: dict[str, str],
    evaluator: Evaluator,
) -> dict[str, list[str]]:
    """
    Runs the inference pipeline on the dataset using the evaluator.

    Args:
        args (argparse.Namespace): Command line arguments.
        dataset (Any): The prepared dataset.
        context (dict[str, str]): Dictionary specifying context.
        evaluator (Evaluator): The initialized model wrapper.
    """
    evaluator_config_file = os.path.join(
        EVALUATOR_ARG_DIR, f"{args.evaluator.replace('/', '_')}.yaml"
    )
    evaluator_args, _, prompt_args = load_args(evaluator_config_file)
    evaluator_args.inference_kwargs.update(
        {
            "eos_token_id": evaluator.tokenizer.eos_token_id,
            "pad_token_id": evaluator.tokenizer.pad_token_id,
        }
    )

    # get task prompts
    evaluation_prompt_path = os.path.join(
        PROMPT_DIR, "evaluation_prompts", "conclusion_evaluation"
    )
    sys_message_file = os.path.join(evaluation_prompt_path, "system_message.txt")
    prompt_file = os.path.join(evaluation_prompt_path, "prompt.txt")
    few_shot_file = os.path.join(evaluation_prompt_path, "few_shot_prompt.txt")

    prompt_manager = PromptManager()
    sys_message, prompt, few_shot_examples = prompt_manager.get_input_prompts(
        sys_message_file_path=sys_message_file,
        prompt_file_path=prompt_file,
        few_shot_prompt_file_path=few_shot_file,
        num_shots=args.num_shots,
    )

    # convert prompts and encode
    output_key = "evaluation_prompt"

    substitution_dict: dict[str, str] = {
        "<model-answer>": "<model-answer>",
        "truth-teller": context["truth-teller"],
        "liar": context["liar"],
    }

    dataset = dataset.map(
        prompt_manager.create_chat_prompt,
        fn_kwargs={
            "input_key": "model_answer",
            "output_key": output_key,
            "user_prompt": prompt,
            "user_prompt_special_token": "<model-answer>",
            "few_shot_examples": few_shot_examples,
            "substitution_dict": substitution_dict,
            "allow_system_message": prompt_args.system_message,
            "system_message": sys_message,
            "tokenizer": evaluator.tokenizer,
        },
        batched=True,
        batch_size=len(dataset),
        load_from_cache_file=False,
    )
    encoded_input = evaluator.tokenizer(
        dataset[output_key], padding=True, return_tensors="pt"
    ).to(args.device)

    # inference
    generated_output = inference_pipeline(
        encoded_input_dict=encoded_input,
        inference_function=evaluator.inference,
        function_kwargs={
            "inference_kwargs": evaluator_args.inference_kwargs,
        },
        batch_size=args.batch_size,
    )

    return generated_output


def parse_conclusions(
    args: argparse.Namespace,
    dataset: Dataset,
    num_characters: int,
    evaluator: ConclusionEvaluator,
) -> dict[str, list[dict[str, bool]]]:
    """
    Parses the conclusions from the given dataset using the provided evaluator.

    Args:
        args (argparse.Namespace): The command line arguments.
        dataset (Dataset): The dataset containing the input data.
        num_characters (int): The number of characters in the dataset.
        evaluator (ConclusionEvaluator): The evaluator to use for parsing the conclusions.

    Returns:
        dict[str, list[str, dict[str, bool]]]: A dictionary containing the extracted conclusions.
            The keys are "model_conclusion" and "gt_conclusion", and the values are lists of dictionaries
            representing the extracted conclusions. Each dictionary has keys representing the characters
            and values representing whether they are truth tellers or liars.
    """
    extracted_conclusions: dict[str, list[dict[str, bool]]] = {
        "model_conclusion": [],
        "gt_conclusion": [],
    }
    missing_extractions: dict[str, list[str]] = {
        "model_answer": [],
        "gt_conclusion": [],
    }
    idx_extracted_conclusions: list[int] = []
    idx_missing_exctractions: list[int] = []

    context = dataset["metadata"][0]["context"]

    # extract conclusion by parsing
    for idx, data_row in enumerate(dataset):
        model_answer = data_row["model_answer"]
        gt_conclusion = data_row["solutions"][0]
        extracted_conclusion = evaluator.rule_based_evaluation(
            model_answer, context, num_characters
        )

        if extracted_conclusion:
            extracted_conclusions["model_conclusion"].append(extracted_conclusion)
            extracted_conclusions["gt_conclusion"].append(gt_conclusion)
            idx_extracted_conclusions.append(idx)
        else:  # failed regex parsing (pass to model)
            missing_extractions["model_answer"].append(model_answer)
            missing_extractions["gt_conclusion"].append(gt_conclusion)
            idx_missing_exctractions.append(idx)

    # extract conclusion using model
    if missing_extractions["model_answer"]:
        missing_extraction_data = Dataset.from_dict(missing_extractions)
        extracted_data = run_evaluator(
            args, missing_extraction_data, context, evaluator
        )

        # parse extractions
        for sample in extracted_data["generated_output"]:
            extracted_conclusion = evaluator.rule_based_evaluation(
                sample, context, num_characters
            )
            extracted_conclusions["model_conclusion"].append(extracted_conclusion)
            extracted_conclusions["gt_conclusion"].append(gt_conclusion)

    # reorder conclusions
    all_indicees = idx_extracted_conclusions + idx_missing_exctractions
    extracted_conclusions["model_conclusion"] = reorder_list(
        extracted_conclusions["model_conclusion"], all_indicees
    )
    extracted_conclusions["gt_conclusion"] = reorder_list(
        extracted_conclusions["gt_conclusion"], all_indicees
    )

    return extracted_conclusions


def process_datasets(
    args: argparse.Namespace,
    model: str,
    strategy: Literal[
        "zero_shot",
        "four_shot",
        "eight_shot",
        "zero_cot",
        "four_cot",
        "eight_cot",
        "cot_sc",
    ],
    configs: dict[str, Any],
    stats: dict[str, dict],
    evaluator: ConclusionEvaluator,
    input_root_dir: str,
    output_root_dir: str,
    save_extracted_conclusions: bool = True,
) -> dict[str, Any]:
    """
    Processes datasets based on configurations and updates the statistics.

    Args:
        args (argparse.Namespace): Command-line arguments.
        configs (dict[str, Any]): Dataset configurations.
        stats (dict[str, dict]): Existing stats.
        evaluator (ConclusionEvaluator): Evaluation module.

    Returns:
        dict[str, Any]: Updated stats.
    """
    for statements in tqdm(
        configs["statement_types"], desc="Processing statement types", leave=False
    ):
        statements_id = f"statements_{''.join(str(s) for s in statements)}"
        stats.setdefault(statements_id, {})

        for characters in configs["characters"]:
            characters_str = f"characters_{characters}"
            stats[statements_id].setdefault(characters_str, {})
            stats[statements_id][characters_str].setdefault(model, {})
            stats[statements_id][characters_str][model].setdefault(strategy, {})

            for run_nr in range(args.num_samples):
                model_answer_subdir = os.path.join(
                    statements_id,
                    characters_str,
                    model,
                    strategy,
                )
                dataset_path = os.path.join(
                    input_root_dir,
                    model_answer_subdir,
                    f"model_answers_{run_nr}.jsonl",
                )

                try:
                    dataset = prepare_dataset_from_disk(dataset_path)
                except Exception as e:
                    logging.warn(e)
                    continue

                conclusion_dict = parse_conclusions(
                    args, dataset, characters, evaluator
                )
                accuracy = compute_accuracy(conclusion_dict)

                stats[statements_id][characters_str][model][strategy][
                    f"sample_{run_nr}"
                ] = {"accuracy": accuracy}

                # save extracted conclusions
                if save_extracted_conclusions:
                    dataset = dataset.add_column(
                        "extracted_conclusion", conclusion_dict["model_conclusion"]
                    )
                    output_path = os.path.join(output_root_dir, model_answer_subdir)
                    write_dataset_to_jsonl(
                        dataset=dataset,
                        file_path=os.path.join(
                            output_path, f"extracted_conclusions_{run_nr}.jsonl"
                        ),
                    )

                # save stats
                evaluator.save_results_to_json(
                    result_dict=stats,
                    file_path=os.path.join(output_root_dir, "stats.json"),
                )

    return stats


def main() -> None:
    """
    Main function to orchestrate the execution flow.
    """
    args = parse_arguments()

    setup_logging(args.verbose)
    set_seed(args.seed)

    # paths
    input_root_dir = os.path.join(MODEL_ANSWER_DIR, args.context)
    output_root_dir = os.path.join(OUTPUT_DIR, args.context)

    # model
    conclusion_evaluator = initialize_evaluator(args)

    # dataset configs
    dataset_configs = read_yaml_file(os.path.join(DATA_ARG_DIR, "dataset_params.yaml"))

    # run inference on dataset
    stats_file = os.path.join(output_root_dir, "stats.json")
    updated_stats = load_existing_stats(stats_file)

    for strategy in args.strategies:
        for model in args.models:
            updated_stats = process_datasets(
                args=args,
                model=model,
                strategy=strategy,
                configs=dataset_configs,
                stats=updated_stats,
                evaluator=conclusion_evaluator,
                save_extracted_conclusions=True,
                input_root_dir=input_root_dir,
                output_root_dir=output_root_dir,
            )


if __name__ == "__main__":
    main()
