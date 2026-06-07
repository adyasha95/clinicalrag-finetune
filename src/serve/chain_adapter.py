"""Drop-in replacement for the Anthropic client used in chain.py.

If FINETUNED_MODEL_URL is set, routes requests to the local vLLM server
using the OpenAI-compatible API. Otherwise falls back to the real
Anthropic API — so the existing RAG system keeps working unchanged.

Usage in chain.py — replace:
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model=MODEL, api_key=..., max_tokens=2048)

With:
    from src.serve.chain_adapter import get_llm
    llm = get_llm()

The returned object is always a LangChain chat model, so the rest of
chain.py requires zero changes.

Environment variables:
    FINETUNED_MODEL_URL   Base URL of the vLLM server, e.g. http://localhost:8000
                          If unset, uses the Anthropic API as normal.
    ANTHROPIC_API_KEY     Required only when falling back to Anthropic.
    FINETUNED_MODEL_NAME  Model name to send to vLLM (default: finetuned-clinical)
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel


_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_MAX_TOKENS = 2048


def get_llm(
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Return the appropriate LangChain chat model based on environment config."""
    finetuned_url = os.environ.get("FINETUNED_MODEL_URL", "").strip()

    if finetuned_url:
        return _build_vllm_client(finetuned_url, max_tokens, temperature)
    else:
        return _build_anthropic_client(max_tokens)


def _build_vllm_client(
    base_url: str,
    max_tokens: int,
    temperature: float,
) -> BaseChatModel:
    """Build a LangChain client pointed at the local vLLM OpenAI-compatible server."""
    from langchain_openai import ChatOpenAI

    model_name = os.environ.get("FINETUNED_MODEL_NAME", "finetuned-clinical")

    # vLLM's OpenAI-compatible endpoint doesn't need a real API key,
    # but the OpenAI client requires a non-empty value.
    return ChatOpenAI(
        model=model_name,
        base_url=f"{base_url.rstrip('/')}/v1",
        api_key="not-needed",
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _build_anthropic_client(max_tokens: int) -> BaseChatModel:
    """Build the standard Anthropic LangChain client."""
    from langchain_anthropic import ChatAnthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set and FINETUNED_MODEL_URL is not configured."
        )

    return ChatAnthropic(
        model=_ANTHROPIC_MODEL,
        api_key=api_key,
        max_tokens=max_tokens,
    )


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    url = os.environ.get("FINETUNED_MODEL_URL")
    if not url:
        print("FINETUNED_MODEL_URL not set — would use Anthropic API.")
        sys.exit(0)

    print(f"Testing connection to fine-tuned model at {url}…")
    llm = get_llm()

    from langchain_core.messages import HumanMessage, SystemMessage

    response = llm.invoke([
        SystemMessage(content="You are a clinical trials assistant."),
        HumanMessage(content="What is a Phase 2 clinical trial?"),
    ])
    print(f"\nResponse:\n{response.content}")
