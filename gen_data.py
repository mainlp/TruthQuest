"""
Generate meta-logical puzzles
"""

import argparse
import logging
import os
import random
from typing import TypeVar

from tqdm import tqdm

from metareasoning.dataprep.data_generator import generate_puzzle
from metareasoning.utils.utils import (
    read_yaml_file,
    save_dicts_as_jsonl,
    setup_logging,
    write_to_text_file,
)

T = TypeVar("T")
ENTITIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
STATEMENT_TYPES = {
    0: "AND",
    1: "OR",
    2: "Implication",
    3: "Equivalence (iff)",
    4: "Self-referential",
    5: "Accusation",
}

DATA_PATH = "data"
PROMPT_PATH = "metareasoning/prompts/reasoning_prompts/few_shot"
YAML_PATH = "metareasoning/dataprep/data_config/dataset_params.yaml"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a meta-logical puzzle.")

    # General configs
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random number generator."
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=1, help="Increase verbosity level."
    )
    parser.add_argument(
        "--from-yaml",
        action="store_true",
        default=False,
        help="Read configs from yaml.",
    )

    # Configs for dataset
    parser.add_argument(
        "--sample-size",
        "-N",
        type=int,
        default=200,
        help="Number of puzzles to generate.",
    )
    parser.add_argument(
        "--few-shot-examples",
        type=int,
        default=0,
        help="Number of few-shot examples to generate.",
    )

    # Configs for puzzle
    parser.add_argument(
        "--characters",
        "-c",
        type=int,
        default=3,
        help="Number of characters in the puzzle.",
    )

    parser.add_argument(
        "--num-solutions",
        type=int,
        nargs="+",
        default=[1],
        help="Allowable number of solutions.",
    )

    parser.add_argument(
        "--statements",
        "-s",
        type=int,
        nargs="+",
        default=[0, 4, 5],
        help="Types of statements included in puzzle.",
    )

    return parser.parse_args()


def assemble_few_shot_example(
    problem_statements: list[str], solution: dict[str, bool]
) -> str:
    """
    Assembles a few-shot example for the meta-logical reasoning task.

    Args:
        problem_statements (list[str]): A list of problem statements.
        solution (dict[str, bool]): A dictionary mapping characters to their identities (True for truth-teller, False for liar).

    Returns:
        str: The assembled few-shot example.
    """
    statements = "\n".join(problem_statements)
    few_shot_example = f"<user>\n{statements}\n</user>\n"

    solution_str = "<assistant>\nREASONING:\n...\n\nCONCLUSION:\n"
    for char, identity in solution.items():
        solution_str += f"{char}: {'truth-teller' if identity else 'liar'}\n"

    few_shot_example += f"{solution_str}</assistant>\n"

    return few_shot_example


def generate_dataset(args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    """
    Generate a dataset of meta-logical puzzles.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        tuple[list[dict], list[str]]: List of puzzle dictionaries and few-shot examples.
    """
    # Info about dataset
    allowed_statements = " - ".join(
        [STATEMENT_TYPES[statement_type] for statement_type in args.statements]
    )
    logging.info(f"Allowed statement types: {allowed_statements}")

    # Generate character names dynamically based on the number of characters
    characters = ENTITIES[: args.characters]
    logging.info(f"Generated characters: {characters}")

    puzzles: list[dict] = []
    puzzle_ids: list[list] = []
    few_shot_examples: list[str] = []

    while len(puzzles) < args.sample_size:
        (
            problem_statements_natural_language,
            problem_statements_fol,
            symbolic_reasoning_path,
            solutions,
            number_solutions,
            statement_types,
        ) = generate_puzzle(
            characters,
            valid_statements=args.statements
        )

        if number_solutions in args.num_solutions:
            if statement_types not in puzzle_ids:
                puzzle_dict = {
                    "problem": problem_statements_natural_language,
                    "problem_logic": problem_statements_fol,
                    "symbolic_reasoning": symbolic_reasoning_path,
                    "solutions": solutions,
                    "metadata": {
                        "puzzle_idx": len(puzzle_ids),
                        "seed": args.seed,
                        "num_characters": args.characters,
                        "statement_types": statement_types,
                    },
                }

                puzzles.append(puzzle_dict)
                puzzle_ids.append(statement_types)

    # generate few-shot examples if specified
    if args.few_shot_examples > 0:
        while len(few_shot_examples) < args.few_shot_examples:
            (
                problem_statements_natural_language,
                problem_statements_fol,
                symbolic_reasoning_path,
                solutions,
                number_solutions,
                statement_types,
            ) = generate_puzzle(
                characters,
                valid_statements=args.statements
            )

            if number_solutions in args.num_solutions:
                if statement_types not in puzzle_ids:
                    few_shot_examples.append(
                        assemble_few_shot_example(
                            problem_statements=problem_statements_natural_language,
                            solution=solutions[0],
                        )
                    )
                    puzzle_ids.append(statement_types)

    return puzzles, few_shot_examples


def main():
    args = parse_arguments()

    # Set up logging based on verbosity
    setup_logging(args.verbose)

    # Set seed if provided
    if args.seed > 0:
        random.seed(args.seed)
        logging.debug(f"Random seed set to {args.seed}")

    # Parse YAML if specified
    if args.from_yaml:
        dataset_configs = read_yaml_file(YAML_PATH)
        statement_types = dataset_configs["statement_types"]
        num_characters = dataset_configs["characters"]
        args.num_solutions = dataset_configs["num_solutions"]
    else:
        statement_types = [args.statements]
        num_characters = [args.characters]

    for statements in tqdm(
        statement_types, desc="Processing statement types", leave=True
    ):
        for characters in num_characters:
            args.statements = statements
            args.characters = characters
            puzzles, few_shot_examples = generate_dataset(args=args)

            # save puzzle
            data_config_path = os.path.join(
                f"statements_{''.join(str(s) for s in statements)}",
                f"characters_{characters}",
            )
            file_path = os.path.join(
                DATA_PATH,
                data_config_path,
                "puzzles.jsonl",
            )
            save_dicts_as_jsonl(puzzles, file_path)

            # save few-shot examples - if any
            if few_shot_examples:
                final_few_shot_text = "---\n".join(few_shot_examples)
                few_shot_file_path = os.path.join(
                    PROMPT_PATH,
                    data_config_path,
                    "few_shot_prompt.txt",
                )
                write_to_text_file(final_few_shot_text, few_shot_file_path)

    logging.info("Puzzle generation complete.")


if __name__ == "__main__":
    main()
