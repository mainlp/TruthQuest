"""
Data class to handle model arguments
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelArgs:
    init_kwargs: dict[str, Any]
    inference_kwargs: dict[str, Any]


@dataclass
class PromptArgs:
    system_message: bool = False
    cautious_mode: bool = False
