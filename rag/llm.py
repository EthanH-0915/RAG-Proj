"""Thin client for a local Ollama server's chat endpoint.

Standard library only (urllib), like config.py's .env reader: the generator is
one HTTP call, not worth a dependency. Calls are deterministic by default
(temperature 0, fixed seed) so an evaluation run can be repeated, and a JSON
schema can be passed as `format`, which Ollama enforces while decoding, so the
reply always parses.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import config


class LLMError(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int
    output_tokens: int
    seconds: float


def chat(
    messages: list[dict],
    schema: dict | None = None,
    *,
    model: str = config.LLM_MODEL,
    temperature: float = 0.0,
    seed: int = 0,
    num_ctx: int = config.LLM_NUM_CTX,
    timeout: float = 900,
) -> ChatResult:
    """One non-streaming chat completion. Raises LLMError with a hint when the
    server is down, the model is missing, or the prompt filled the window."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx},
    }
    if schema is not None:
        body["format"] = schema

    request = urllib.request.Request(
        f"{config.LLM_URL}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            reply = json.loads(response.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        if e.code == 404:
            raise LLMError(f"{detail}\nPull the model first: ollama pull {model}") from e
        raise LLMError(f"Ollama returned HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(
            f"Cannot reach Ollama at {config.LLM_URL} ({e.reason}). Is it running?"
        ) from e

    prompt_tokens = reply.get("prompt_eval_count", 0)
    # Ollama truncates an over-long prompt instead of failing; a prompt that
    # used the whole window almost certainly lost its start.
    if prompt_tokens >= num_ctx:
        raise LLMError(
            f"Prompt filled the {num_ctx}-token window and was truncated; "
            "send fewer or shorter sources, or raise LLM_NUM_CTX."
        )
    return ChatResult(
        content=reply["message"]["content"],
        prompt_tokens=prompt_tokens,
        output_tokens=reply.get("eval_count", 0),
        seconds=time.perf_counter() - start,
    )
