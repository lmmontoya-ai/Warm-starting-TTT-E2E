"""Pinned, local prompt rendering for the T1a short-context battery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


PROMPT_RENDERER_VERSION = "tier1_prompt_v1"


@dataclass(frozen=True)
class RenderedExample:
    example_id: str
    prompt: str
    choices: tuple[str, ...]
    gold: int


def _clean_hellaswag(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", str(text))
    return text.replace("  ", " ").strip()


def render_example(
    task: str, row: dict[str, Any], *, row_index: int
) -> RenderedExample:
    if task == "hellaswag":
        ctx_a = str(row.get("ctx_a", "")).strip()
        ctx_b = str(row.get("ctx_b", "")).strip()
        if ctx_b:
            ctx_b = ctx_b[0].upper() + ctx_b[1:]
        prompt = _clean_hellaswag(f"{ctx_a} {ctx_b}")
        choices = tuple(" " + _clean_hellaswag(x) for x in row["endings"])
        gold = int(row["label"])
        example_id = str(row.get("ind", row.get("source_id", row_index)))
    elif task == "arc_easy":
        prompt = f"Question: {str(row['question']).strip()}\nAnswer:"
        choice_block = row["choices"]
        labels = [str(x) for x in choice_block["label"]]
        choices = tuple(" " + str(x).strip() for x in choice_block["text"])
        answer_key = str(row["answerKey"])
        if answer_key not in labels:
            raise ValueError(f"ARC answer {answer_key!r} absent from labels {labels}")
        gold = labels.index(answer_key)
        example_id = str(row.get("id", row_index))
    elif task == "piqa":
        prompt = f"Question: {str(row['goal']).strip()}\nAnswer:"
        choices = (" " + str(row["sol1"]).strip(), " " + str(row["sol2"]).strip())
        gold = int(row["label"])
        example_id = str(row.get("id", row_index))
    elif task == "winogrande":
        sentence = str(row["sentence"])
        if sentence.count("_") != 1:
            raise ValueError("WinoGrande sentence must contain exactly one blank")
        prefix, suffix = sentence.split("_", 1)
        prompt = prefix
        choices = (
            str(row["option1"]) + suffix,
            str(row["option2"]) + suffix,
        )
        gold = int(row["answer"]) - 1
        example_id = str(row.get("id", row_index))
    else:
        raise KeyError(f"Unsupported T1a task: {task}")
    if not choices or gold < 0 or gold >= len(choices):
        raise ValueError(f"Invalid rendered example for {task}/{example_id}")
    return RenderedExample(
        example_id=example_id, prompt=prompt, choices=choices, gold=gold
    )


def encode_prompt_choice(
    tokenizer, prompt: str, choice: str, *, bos_token_id: int
) -> tuple[list[int], int]:
    # Moving trailing whitespace to the continuation follows causal-LM harness
    # semantics and avoids tokenizer-boundary inconsistencies.
    trailing = prompt[len(prompt.rstrip()) :]
    prompt = prompt.rstrip()
    choice = trailing + choice
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    whole_ids = list(tokenizer.encode(prompt + choice, add_special_tokens=False))
    common = 0
    for left, right in zip(prompt_ids, whole_ids):
        if left != right:
            break
        common += 1
    context_ids = whole_ids[:common]
    continuation_ids = whole_ids[common:]
    if not continuation_ids:
        raise ValueError("Choice produced no continuation tokens")
    tokens = [int(bos_token_id), *map(int, context_ids), *map(int, continuation_ids)]
    choice_start = 1 + len(context_ids)
    return tokens, choice_start
