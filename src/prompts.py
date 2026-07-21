"""Zero-shot and few-shot prompt construction for LLM slot filling.

The prompts are designed to:
  * Explain the task in plain English
  * List every BIO tag in the dataset
  * Show the exact input sentence and request a strict, machine-parseable
    output format
  * (few-shot) Include k labeled examples drawn from the TRAINING set
"""
from __future__ import annotations

from typing import List, Sequence


SYSTEM_PROMPT = (
    "You are an expert at English sequence labelling for airline-travel "
    "queries (ATIS slot filling). Given a sentence, you assign a BIO tag "
    "to every word. Output ONLY a code-fenced list of BIO tags, one tag per "
    "line, in the same order as the words in the sentence. Do not output "
    "any other text."
)


def render_tag_inventory(all_tags: Sequence[str]) -> str:
    """Render the BIO tag inventory as a sorted, alphabetised list."""
    real = sorted(t for t in all_tags if t not in ("PAD", "X"))
    lines = [f"  - {t}" for t in real]
    return "Available BIO tags (use exactly one per word):\n" + "\n".join(lines)


def render_few_shot_examples(examples: Sequence[dict]) -> str:
    """Render k labeled examples in the format:

        Sentence: <space-joined tokens>
        Tags:
        ```
        <tag>
        <tag>
        ...
        ```
    """
    out = []
    for i, ex in enumerate(examples, 1):
        sentence = " ".join(ex["tokens"])
        tags_block = "\n".join(ex["tags"])
        out.append(
            f"Example {i}:\n"
            f"Sentence: {sentence}\n"
            f"Tags:\n```\n{tags_block}\n```"
        )
    return "\n\n".join(out)


def build_zero_shot_prompt(sentence: str, all_tags: Sequence[str]) -> str:
    """Zero-shot user prompt for a single sentence."""
    return (
        "TASK\n"
        "Assign a BIO slot-filling tag to every word of the input sentence. "
        "B- begins a slot, I- continues the same slot, O is outside any slot.\n\n"
        f"{render_tag_inventory(all_tags)}\n\n"
        f"INPUT SENTENCE\n{sentence}\n\n"
        "OUTPUT\n"
        "Return a fenced code block with exactly one BIO tag per line, in word order. "
        "Use the format:\n```\n<tag-1>\n<tag-2>\n...\n```\n"
        "Do not include any words, comments, or extra text."
    )


def build_few_shot_prompt(
    sentence: str, all_tags: Sequence[str], examples: Sequence[dict]
) -> str:
    """Few-shot user prompt: zero-shot prompt + k labeled examples before the target."""
    examples_block = render_few_shot_examples(examples)
    return (
        f"TASK\n"
        f"Assign a BIO slot-filling tag to every word of the input sentence. "
        f"B- begins a slot, I- continues the same slot, O is outside any slot.\n\n"
        f"{render_tag_inventory(all_tags)}\n\n"
        f"EXAMPLES (correctly labeled)\n{examples_block}\n\n"
        f"INPUT SENTENCE (your turn)\n{sentence}\n\n"
        f"OUTPUT\n"
        f"Return a fenced code block with exactly one BIO tag per line, in word order. "
        f"Use the format:\n```\n<tag-1>\n<tag-2>\n...\n```\n"
        f"Do not include any words, comments, or extra text."
    )
