"""
Code to generate data for meta-logical reasoning.
"""

import random
from itertools import product
from typing import List, TypeVar

from sympy import And, Equivalent, Implies, Not, Or, Symbol, symbols
from sympy.core.relational import Equality, Unequality
from sympy.logic.boolalg import (
    BooleanFalse,
    BooleanTrue,
    distribute_or_over_and,
    to_dnf,
)

T = TypeVar("T")


def random_k_sublist(set_: List[T], k: int) -> List[T]:
    """Generate a random sublist of length k from a given list set_.

    Args:
    set_ (List[T]): A list from which elements are to be chosen.
    k (int): Number of elements to choose.

    Returns:
    List[T]: A random sublist of length k.
    """
    return random.sample(set_, k)


def random_permut(k: int) -> List[int]:
    """Generate a random permutation of integers from 1 to k.

    Args:
    k (int): The upper limit of the range (inclusive).

    Returns:
    List[int]: A random permutation of integers from 1 to k.
    """
    perm = list(range(1, k + 1))
    random.shuffle(perm)
    return perm


def inverse_permut(permut: List[int]) -> List[int]:
    """Find the inverse of a given permutation.

    Args:
    permut (List[int]): A permutation of integers.

    Returns:
    List[int]: The inverse permutation.
    """
    inverse = [0] * len(permut)
    for i, p in enumerate(permut):
        inverse[p - 1] = i + 1
    return inverse


def id(kk: int, x: bool) -> bool:
    """Perform a conditional logical operation based on the value of kk.

    Args:
    kk (int): A condition variable.
    x (bool): A boolean value.

    Returns:
    bool: The negation of x if kk equals 0, else x itself.
    """
    return not x if kk == 0 else x


def sample_k_variants(list_: List[T], k: int) -> List[List[T]]:
    """Generate k-combinations (variations) of a given list.

    Args:
    list_ (List[T]): The original list to generate combinations from.
    k (int): The length of each combination.

    Returns:
    List[List[T]]: A list of all k-combinations of the original list.
    """
    if k == 0:
        return [[]]
    if k == 1:
        return [[item] for item in list_]
    return [[item] + rest for item in list_ for rest in sample_k_variants(list_, k - 1)]


def sample_statement(
    speaker_id: int, statement_type: int, n: int
) -> tuple[int, int, list[int], list[int]]:
    """Create a random statement for the logic puzzle, including self-referential and comparative statements.

    Statement types map to:
        0: logical AND
        1: logical OR
        2: logical implication (if ... then)
        3: logical equivalence (iff)
        4: self-referential ("I am a truth-teller")
        5: accusation ("X is a truth-teller/liar")

    Args:
    speaker_id (int): Index of the character making the statement, excluded from some statement types.
    statement_type (int): Type of statement.
    n (int): Total number of characters.

    Returns:
    tuple[int, int, list[int], list[int]]: A tuple representing:
        [speaker_id, statement_type, ]
    """
    assert statement_type <= 5, "Please choose a valid statement_type: 0-5!"

    if statement_type < 4:
        chars = random_k_sublist([x for x in range(n) if x != speaker_id], 2)
        truths = random.choices([0, 1], k=2)
        return speaker_id, statement_type, chars, truths

    # Self-referential statement ("I am a truth-teller")
    elif statement_type == 4:
        return speaker_id, statement_type, [speaker_id], [1]

    # A character making a claim about another character's truthfulness
    else:
        other_char = random.choice([x for x in range(n) if x != speaker_id])
        return speaker_id, statement_type, [other_char], [random.choice([0, 1])]


def generate_statement_components(
    statement: tuple[int, int, list[int], list[int]], characters: list[str]
) -> tuple[list[str], list[str], list[str], str, str]:
    """
    Generate the base components of a statement for both symbolic and English translations.

    Args:
        statement (tuple[int, int, list[int], list[int]]): The statement data including speaker_id, st_type, subjects, and truths.
        characters (list[str]): List of character names.

    Returns:
        tuple[list[str], list[str], str, str]: A tuple containing the characters involved in the statement,
                                               their corresponding truth claims, the operation symbol,
                                               and the operation word.
    """
    _, st_type, chars, truths = statement
    char_names = [characters[char] for char in chars]

    truth_claims = [("¬" if truth == 0 else "") for truth in truths]
    truthteller_or_liar = [
        ("a liar" if truth == 0 else "a truth-teller") for truth in truths
    ]

    # Define symbols and words for logical operations
    symbols = ["∧", "v", "→", "↔", "", ""]
    words = ["and", "or", "if...then", "if and only if", "", ""]
    symbol = symbols[st_type]
    word = words[st_type]

    return char_names, truth_claims, truthteller_or_liar, symbol, word


def symbolic_expression(
    statement: tuple[int, int, list[int], list[int]],
    characters: list[str],
) -> str:
    """
    Create a symbolic expression from a statement, indicating which character is making the statement.

    Args:
        statement (tuple[int, int, list[int], list[int]]): A statement consisting of a type, subjects, and their truths.
        characters (list[str]): List of character names.

    Returns:
        str: The symbolic expression, with clear indication of the speaking character.
    """
    speaker_id, st_type, _, _ = statement
    char_making_statement = characters[speaker_id]
    char_names, truth_claims, _, operation, _ = generate_statement_components(
        statement, characters
    )

    if st_type < 4:
        expr = f"{char_making_statement}: {truth_claims[0]} {char_names[0]} {operation} {truth_claims[1]} {char_names[1]}"
        return f"{expr}"

    elif st_type == 4 or st_type == 5:
        return f"{char_making_statement}: {truth_claims[0]} {char_names[0]}"

    return "Invalid statement format"


def natural_language_expression(
    statement: tuple[int, int, list[int], list[int]],
    characters: list[str],
) -> str:
    """
    Translate a puzzle statement into English, clearly denoting which character is making the statement.

    Args:
        statement (tuple[int, int, list[int], list[int]]): A statement consisting of a type, subjects, and their truths.
        characters (list[str]): List of character names involved in the statements.

    Returns:
        str: The translated statement in English, clearly indicating the speaking character.
    """
    speaker, st_type, _, _ = statement
    char_making_statement = characters[speaker]
    char_names, _, truthteller_or_liar, _, word = generate_statement_components(
        statement, characters
    )

    if st_type < 4:
        if st_type == 2:
            expr = f"If {char_names[0]} is {truthteller_or_liar[0]}, then {char_names[1]} is {truthteller_or_liar[1]}."
        else:
            expr = f"{char_names[0]} is {truthteller_or_liar[0]} {word} {char_names[1]} is {truthteller_or_liar[1]}."
        return f"{char_making_statement}: {expr}"

    elif st_type == 4:  # Self-referential statement
        return f"{char_making_statement}: I am a truth-teller."

    elif st_type == 5:  # Statement about another character
        return f"{char_making_statement}: {char_names[0]} is {truthteller_or_liar[0]}."

    return "Invalid statement"


def generate_symbolic_statements(
    characters: list[str], statements: list[tuple[int, int, list[int], list[int]]]
) -> And:
    """
    Generate symbolic representations for each character's statement.

    Args:
        characters (List[str]): List of character names.
        statements (List[Statement]): List of statements in the form (speaker_id, statement_type, statement_chars, statement_truths).

    Returns:
        SymbolicStatements: A dictionary mapping each character to their symbolic statement.
    """
    sym_chars = symbols(" ".join(characters))
    sym_statements = []

    for statement in statements:
        speaker, st_type, chars, truths = statement
        speaker_sym = sym_chars[speaker]
        char_syms = [sym_chars[char] for char in chars]
        truth_syms = [
            (Not(char_sym) if truth == 0 else char_sym)
            for char_sym, truth in zip(char_syms, truths)
        ]

        if st_type == 0:  # AND
            sym_statements.append(Equivalent(speaker_sym, And(*truth_syms)))
        elif st_type == 1:  # OR
            sym_statements.append(Equivalent(speaker_sym, Or(*truth_syms)))
        elif st_type == 2:  # IMPLIES
            sym_statements.append(Equivalent(speaker_sym, Implies(*truth_syms)))
        elif st_type == 3:  # EQUIVALENT
            sym_statements.append(Equivalent(speaker_sym, Equivalent(*truth_syms)))
        elif st_type == 4:  # Self-reference (truth-teller)
            sym_statements.append(Equivalent(speaker_sym, speaker_sym))
        elif st_type == 5:  # Claim about another
            sym_statements.append(Equivalent(speaker_sym, truth_syms[0]))

    sym_statement = And(*sym_statements)

    return sym_statement


def sympy_to_custom_str(expr):
    """
    Recursively translate a SymPy logical expression into a custom string format with logical operators.

    Args:
        expr: A SymPy expression.

    Returns:
        str: The translated string representation of the expression.
    """
    if isinstance(expr, BooleanFalse):
        return str(expr)
    elif isinstance(expr, BooleanTrue):
        return str(expr)
    elif isinstance(expr, Symbol):
        return str(expr)
    elif isinstance(expr, And):
        return " ∧ ".join(sympy_to_custom_str(arg) for arg in expr.args)
    elif isinstance(expr, Or):
        return "(" + " v ".join(sympy_to_custom_str(arg) for arg in expr.args) + ")"
    elif isinstance(expr, Not):
        inner_expr = sympy_to_custom_str(expr.args[0])
        return f"¬({inner_expr})" if len(inner_expr) > 1 else f"¬{inner_expr}"
    else:
        inner_left_expr = sympy_to_custom_str(expr.args[0])
        left_expr = (
            f"{inner_left_expr}"
            if len(inner_left_expr) == 1
            or (len(inner_left_expr) == 2 and inner_left_expr[0] == "¬")
            else f"({inner_left_expr})"
        )
        inner_right_expr = sympy_to_custom_str(expr.args[1])
        right_expr = (
            f"{inner_right_expr}"
            if len(inner_right_expr) == 1
            or (len(inner_right_expr) == 2 and inner_right_expr[0] == "¬")
            else f"({inner_right_expr})"
        )
        if isinstance(expr, Implies):
            return "(" + f"{left_expr} → {right_expr}" + ")"
        elif isinstance(expr, Equivalent):
            return "(" + f"{left_expr} ↔ {right_expr}" + ")"
        elif isinstance(expr, Equality):
            return f"{left_expr} = {right_expr}"
        elif isinstance(expr, Unequality):
            return f"{left_expr} != {right_expr}"
        else:
            return str(expr)


def apply_de_morgans(expr):
    """
    Apply De Morgan's law to the given logical expression.

    Parameters:
    expr (object): The logical expression to be transformed.

    Returns:
    object: The transformed logical expression.
    """
    if isinstance(expr, Not):
        if isinstance(expr.args[0], And):
            return Or(Not(expr.args[0].args[0]), Not(expr.args[0].args[1]))
        elif isinstance(expr.args[0], Or):
            return And(Not(expr.args[0].args[0]), Not(expr.args[0].args[1]))
    return expr


def step_by_step_simplification(expr):
    steps = []
    current_expr = expr
    steps.append(sympy_to_custom_str(current_expr))

    # Step 1: Convert equivalences
    new_expr = current_expr.replace(
        lambda expr: isinstance(expr, Equivalent),
        lambda expr: Or(
            And(expr.args[0], expr.args[1]), And(Not(expr.args[1]), Not(expr.args[0]))
        ),
    )
    if new_expr != current_expr:
        current_expr = new_expr
        steps.append(sympy_to_custom_str(current_expr))

    # Step 2: Convert implications
    new_expr = current_expr.replace(
        lambda expr: isinstance(expr, Implies),
        lambda expr: Or(Not(expr.args[0]), expr.args[1]),
    )
    if new_expr != current_expr:
        current_expr = new_expr
        steps.append(sympy_to_custom_str(current_expr))

    # Step 3: Simplify expression using distributive laws and De Morgan's laws iteratively
    changed = True
    while changed:
        changed = False
        # Apply De Morgan's laws
        new_expr = apply_de_morgans(current_expr)
        if new_expr != current_expr:
            current_expr = new_expr
            steps.append(sympy_to_custom_str(current_expr))
            changed = True

        # Distribute OR over AND (a v (b & c)) -> ((a v b) & (a v c))
        new_expr = distribute_or_over_and(current_expr)
        if new_expr != current_expr:
            current_expr = new_expr
            steps.append(sympy_to_custom_str(current_expr))
            changed = True

        # Convert to CNF and simplify
        new_expr = to_dnf(current_expr, simplify=True)
        if new_expr != current_expr:
            current_expr = new_expr
            steps.append(sympy_to_custom_str(current_expr))
            changed = True

    return steps


def gen_reasoning_path(
    statements: list[tuple[int, int, list[int], list[int]]], characters: List[str]
) -> tuple[str, str]:
    """
    Generate a step-by-step reasoning path from the statements to the valid solution specified in the assignment dict.

    Args:
        statements (List[Statement]): Statements of the form (speaker, statement_type, statement_chars, statement_truths).
        characters (List[str]): List of character names.

    Returns:
        str: The full reasoning path.
    """
    symbolic_statements = generate_symbolic_statements(characters, statements)
    reasoning_path = step_by_step_simplification(symbolic_statements)
    return "\n".join(reasoning_path), reasoning_path[-1]


def parse_solution(
    final_reasoning: str, characters: list[str]
) -> list[dict[str, bool]]:
    """
    Parse the final reasoning of a logical puzzle solution into a list of possible solutions.

    Each solution is represented as a dictionary where keys are character names and
    values are booleans indicating the truthfulness (True for truth-teller, False for liar).

    This function also accounts for missing characters in the reasoning and generates all possible
    combinations of solutions for these missing characters.

    Args:
        final_reasoning (str): A string representation of the final reasoning in a logical puzzle,
                               using '∧' for AND, 'v' for OR, and '¬' for NOT.
        characters (list[str]): A list of all characters that should be considered in the solutions.

    Returns:
        list[dict[str, bool]]: A list of dictionaries representing possible solutions.
    """
    clean_reasoning = final_reasoning.translate(str.maketrans("", "", "() "))

    # Split the reasoning into separate solutions based on the 'v' (OR) operator
    solutions = clean_reasoning.split("v")

    final_solutions = []
    for solution in solutions:
        assignments = solution.split("∧")

        # Create a dictionary for each solution by parsing character assignments
        solution_dict = {
            assignment.strip("¬")[-1]: "¬" not in assignment
            for assignment in assignments
        }

        # Determine which characters are missing from the current solution
        missing_characters = set(characters) - set(solution_dict.keys())

        # If there are missing characters, generate all combinations of truth values for them
        if missing_characters:
            for combination in product([True, False], repeat=len(missing_characters)):
                temp_solution = solution_dict.copy()
                temp_solution.update(dict(zip(missing_characters, combination)))
                final_solutions.append(dict(sorted(temp_solution.items())))
        else:
            final_solutions.append(dict(sorted(solution_dict.items())))

    return final_solutions


def generate_puzzle(
    characters: list[str],
    valid_statements: list[int] = [0, 4, 5]
) -> tuple[list[str], list[str], str, list[dict[str, bool]], int, list]:
    """
    Generate a logic puzzle and find all solutions associated to it.

    Args:
        characters (List[str]): List of character names.
        valid_statements (list[int], optional): List of valid statement types.

    Returns:
        Tuple: A tuple containing the puzzle's difficulty, logical expressions,
               solution (if exists), translated statements, and symbolic expressions.
    """
    assert (
        len(valid_statements) >= 1
    ), f"Please specify valid statement types! Currently you have: {valid_statements}"
    n = len(characters)

    # Generate random statements for the puzzle
    puzzle_id = []
    statements = []

    for speaker_id in range(n):
        statement_type = random.choice(valid_statements)
        statement = sample_statement(speaker_id, statement_type, n)
        statements.append(statement)
        puzzle_id.append(
            {
                "speaker_id": statement[0],
                "statement_type": statement[1],
                "chars": statement[2],
                "truths": statement[3],
            }
        )

    # Translate the statements into English
    nl_statements = [
        natural_language_expression(statement, characters) for statement in statements
    ]

    # Create symbolic expressions for each statement
    symbolic_expressions = [
        symbolic_expression(statement, characters) for statement in statements
    ]

    # Reason about problem
    symbolic_reasoning_path, final_reasoning = gen_reasoning_path(
        statements, characters
    )

    if final_reasoning == "True":
        solutions = [
            {char: value for char, value in zip(characters, combination)}
            for combination in product([True, False], repeat=len(characters))
        ]
    elif final_reasoning == "False":
        solutions = []
    else:
        solutions = parse_solution(final_reasoning, characters)

    # Calculate the puzzle's difficulty based on the number of valid solutions
    difficulty = len(solutions)

    return (
        nl_statements,
        symbolic_expressions,
        symbolic_reasoning_path,
        solutions,
        difficulty,
        puzzle_id,
    )
