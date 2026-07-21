"""OpenAI-compatible LLM client (LM Studio) + BIO-list parser.

Exposes:
    LMStudioClient   - thin wrapper around openai.OpenAI for LM Studio
    parse_bio_list   - extract a list of BIO tag strings from free-form LLM output
"""
from __future__ import annotations

import re
from typing import List, Optional

from openai import OpenAI

from .config import LLM_BASE_URL, LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT


class LMStudioClient:
    """OpenAI-compatible client pointed at an LM Studio server.

    LM Studio exposes the OpenAI /v1/chat/completions endpoint, so we can
    use the openai SDK directly. No API key required for local servers.
    """

    def __init__(
        self,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: float = LLM_TIMEOUT,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.client = OpenAI(base_url=base_url, api_key="lm-studio", timeout=timeout)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        """Single, stateless chat completion. No conversation memory."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""


_BIO_LINE_RE = re.compile(r"^[BIOX][-_a-zA-Z0-9.]*$")


def parse_bio_list(text: str, n_tokens: int) -> List[str]:
    """Extract a list of BIO tag strings from free-form LLM output.

    The LLM is asked to return one tag per line, in token order, in a
    code-fenced block. This parser:
      1. prefers a ``` block (```...```), falls back to the whole text
      2. splits on newlines/commas, strips whitespace
      3. validates that each token looks like a BIO tag (starts with B/I/O/X
         and contains only letters/digits/dots/underscores/dashes)
      4. trims / pads to exactly n_tokens

    Returns a list of length n_tokens. Tags that fail validation become 'O'.
    """
    text = text.strip()

    m = re.search(r"```(?:[a-zA-Z]*\n)?(.*?)```", text, re.DOTALL)
    if m:
        body = m.group(1)
    else:
        body = text

    body = body.replace("`", "")
    raw_tokens = re.split(r"[\n,;]+", body)
    candidates: List[str] = []
    for tok in raw_tokens:
        tok = tok.strip().strip("[]\"' ")
        if not tok:
            continue
        if _BIO_LINE_RE.match(tok):
            candidates.append(tok)
        else:
            for piece in tok.split():
                piece = piece.strip("[]\"',")
                if _BIO_LINE_RE.match(piece):
                    candidates.append(piece)

    if len(candidates) >= n_tokens:
        return candidates[:n_tokens]
    while len(candidates) < n_tokens:
        candidates.append("O")
    return candidates


def sanitize_model_name(name: str) -> str:
    """Make a model id safe for use as a filename component.

    Replaces '/' with '_' and strips anything that is not alphanumeric,
    '.', '-', or '_'.  Example: 'google/gemma-3-4b-it' -> 'google_gemma-3-4b-it'.
    """
    s = name.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    return s
