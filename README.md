<h1 align="center">Liar, Liar, Logical Mire: A Benchmark for Suppositional Reasoning in Large Language Models</h1>

<div style="text-align: center; width: 100%;">
  <!-- Container to align the image and the caption -->
  <div style="display: inline-block; text-align: left; width: 85%;">
    <img src="assets/images/knights_and_knaves.png" style="width: 100%;" alt="knights and knaves">
    <p style="color: gray; font-size: small; margin: 0;">
      <em>Generated by DALL·E 3</em>
    </p>
  </div>
</div>

<br>


Knights and knaves problems represent a classic genre of logical puzzles where characters either tell the truth or lie. The objective is to logically deduce each character's identity based on their statements. The challenge arises from the truth-telling or lying behavior, which influences the logical implications of each statement. Solving these puzzles requires not only direct deductions from individual statements, but the ability to assess the truthfulness of statements by reasoning through various hypothetical scenarios. As such, knights and knaves puzzles serve as compelling examples of suppositional reasoning. In this paper, we introduce \emph{TruthQuest}, a benchmark for suppositional reasoning based on the principles of knights and knaves puzzles. Our benchmark presents problems of varying complexity, considering both the number of characters and the types of logical statements involved. Evaluations on \emph{TruthQuest} show that large language models like Llama 3 and Mixtral-8x7B exhibit significant difficulties solving these tasks. A detailed error analysis of the models' output reveals that lower-performing models exhibit a diverse range of reasoning errors, frequently failing to grasp the concept of truth and lies. In comparison, more proficient models primarily struggle with accurately inferring the logical implications of potentially false statements.


## Table of Contents
- [Setup](#setup)
- [Generate Puzzles](#generate-puzzles)
- [Run Models](#run-models)
- [Evaluate Performance](#evaluate-performance)
- [LLM-Based and Human Annotations](#llm-based-and-human-annotations)
- [Human-Annotated CoT Prompts](#human-annotated-cot-prompts)
- [License](#license)
- [Citation](#citation)

## Setup
All code was developed and tested on Ubuntu 22.04 with Python 3.11

To run the current code, we recommend to use Poetry:
```bash
poetry install                          # Install dependencies
poetry shell                            # Activate virtual environment
# Work for a while
deactivate
```

Please make sure to configure your [HuggingFace](https://huggingface.co/) credentials to download respective models.

## Generate Puzzles
To generate puzzles, run the following command:
```
python gen_data.py --from-yaml
```
This will generate the corresponding data. Alternatively, you can fetch the data from [here](https://huggingface.co/datasets/mainlp/TruthQuest).

## Run Models
To run models, run the following command:
```
python run.py --model <hf-model-name>
```

In this project, we used the following models:

- [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
- [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
- [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

Note that in order to use a new model, you need to add a configuration file in this [folder](metareasoning/models/model_config).

## Evaluate Performance
To evaluate the performance of the models, run the following command:
```
python evaluate_conclusion.py
```

For specific command-line arguments, please refer to the code.

## LLM-Based and Human Annotations
We publish all [LLM-based](https://huggingface.co/datasets/mainlp/TruthQuest-AI-Annotations) and [human annotations](https://huggingface.co/datasets/mainlp/TruthQuest-Human-Annotations) in our respective HuggingFace data repository. The TruthQuest dataset can be found [here](https://huggingface.co/datasets/mainlp/TruthQuest).

## Human-Annotated CoT Prompts
We provide up to 8 human-annotated CoT examples for each dataset configuration. Please see [this folder](metareasoning/prompts/reasoning_prompts/cot/) for further information.

## License
[![MIT license](https://img.shields.io/badge/License-Creative%20Commons%20Attribution--ShareAlike%204.0%20International%20Public%20License-green.svg)](https://creativecommons.org/licenses/by-sa/4.0)

This work is licensed under a [CC BY-SA 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/).

## Citation

If you find our work helpful, you can cite this paper as:
```
@inproceedings{mondorf-plank-2024-liar,
    title = "Liar, Liar, Logical Mire: A Benchmark for Suppositional Reasoning in Large Language Models",
    author = "Mondorf, Philipp  and Plank, Barbara",
    editor = "Al-Onaizan, Yaser  and Bansal, Mohit  and Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.404",
    pages = "7114--7137",
}
```