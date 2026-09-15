import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

from rag.config import (
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_REASONING,
    LLM_TEMPERATURE,
)

load_dotenv()


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    key = os.environ.get("AICREDITS_API_KEY")
    if not key:
        raise RuntimeError("AICREDITS_API_KEY is not set — add it to .env")
    return OpenAI(base_url=LLM_BASE_URL, api_key=key)


def complete(system: str, user: str, reasoning: bool = LLM_REASONING) -> str:
    """One turn against the LLM. Reasoning stays off unless asked for."""
    return (
        get_client()
        .chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=LLM_TEMPERATURE,
            # A token cap with reasoning on can be swallowed whole by the
            # reasoning trace, returning empty content that you still pay for.
            max_tokens=None if reasoning else LLM_MAX_TOKENS,
            extra_body={"reasoning": {"enabled": reasoning}},
        )
        .choices[0]
        .message.content
        or ""
    )
