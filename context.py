"""
context.py — token budget tracking and message history compression.

Responsibilities:
  - Estimate how many tokens the current message history consumes
  - Track pressure against the model's context window limit
  - Compress old messages via sliding window when pressure gets too high
  - Summarise a completed task's messages into a single summary message
"""

import json
import os
from dataclasses import dataclass

import anthropic

from prompts import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Context window limits per model
# ---------------------------------------------------------------------------

MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # Anthropic
    "claude-sonnet-4-5":       200_000,
    "claude-opus-4-5":         200_000,
    "claude-haiku-4-5":        200_000,
    # Gemini
    "gemini-2.0-flash":      1_000_000,
    "gemini-1.5-flash":      1_000_000,
    "gemini-1.5-pro":        2_000_000,
}

# How many tokens to keep free for the model's response
RESPONSE_RESERVE_TOKENS = 4_096

# Trigger compression when this fraction of the budget is consumed
COMPRESSION_THRESHOLD = 0.70

# How many of the most recent messages to keep verbatim during compression
KEEP_LAST_N_MESSAGES = 6


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

@dataclass
class TokenBudget:
    """Tracks token usage against a model's context window."""
    model: str
    used: int = 0

    @property
    def limit(self) -> int:
        return MODEL_CONTEXT_LIMITS.get(self.model, 128_000)

    @property
    def available(self) -> int:
        return self.limit - self.used - RESPONSE_RESERVE_TOKENS

    @property
    def pressure(self) -> float:
        """0.0 = empty window, 1.0 = fully consumed."""
        usable = self.limit - RESPONSE_RESERVE_TOKENS
        return self.used / usable if usable > 0 else 1.0

    @property
    def needs_compression(self) -> bool:
        return self.pressure >= COMPRESSION_THRESHOLD

    def __str__(self) -> str:
        return (
            f"tokens: {self.used:,} / {self.limit:,} "
            f"({self.pressure:.0%} used, {self.available:,} remaining)"
        )


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def estimate_tokens(messages: list[dict], system_prompt: str = SYSTEM_PROMPT) -> int:
    """Rough token estimate: 1 token per 4 characters of JSON.

    Fast and free — no API call needed. Used for continuous monitoring.
    Slightly underestimates (real tokenizers are more complex) but good
    enough to decide when to trigger compression.
    """
    content = json.dumps(messages) + system_prompt
    return len(content) // 4


def count_tokens_anthropic(
    messages: list[dict],
    tools: list[dict],
    model: str,
) -> int:
    """Exact token count via Anthropic's count_tokens API.

    Makes a real API call (no inference, no cost beyond the count call itself).
    Use this for accurate budget tracking at key checkpoints.
    """
    # Strip provider-specific fields Anthropic doesn't accept
    clean_messages = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            clean_content = [
                {k: v for k, v in block.items() if k != "tool_name"}
                if isinstance(block, dict) and block.get("type") == "tool_result"
                else block
                for block in content
            ]
            clean_messages.append({"role": msg["role"], "content": clean_content})
        else:
            clean_messages.append(msg)

    client = anthropic.Anthropic()
    response = client.messages.count_tokens(
        model=model,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=clean_messages,
    )
    return response.input_tokens


# ---------------------------------------------------------------------------
# Sliding window compression
# ---------------------------------------------------------------------------

def compress_history(
    messages: list[dict],
    keep_last_n: int = KEEP_LAST_N_MESSAGES,
) -> list[dict]:
    """Trim message history by keeping the first message (user task) and
    the last N messages verbatim, dropping everything in between.

    The first message (the original task) is always preserved — it's the
    agent's north star. Without it, the agent loses track of what it's doing.

    Before compression (10 messages):
        [0] user: original task
        [1] assistant: tool call
        [2] user: tool result
        [3] assistant: tool call
        [4] user: tool result
        ...
        [9] assistant: tool call  ← keep (within last N)

    After compression (keep_last_n=4):
        [0] user: original task    ← always kept
        [1] user: [TRIMMED ...]    ← marker so agent knows history was cut
        [6] user: tool result      ← last N messages
        [7] assistant: tool call
        [8] user: tool result
        [9] assistant: tool call
    """
    if len(messages) <= keep_last_n + 1:
        # Nothing to compress — already short enough
        return messages

    first = messages[0]          # original task — always keep
    recent = messages[-keep_last_n:]  # most recent N messages

    dropped = len(messages) - 1 - keep_last_n
    trim_marker = {
        "role": "user",
        "content": (
            f"[CONTEXT TRIMMED: {dropped} earlier messages were removed to stay within "
            f"the context window. The current task context and recent messages follow.]"
        ),
    }

    return [first, trim_marker] + recent


def compress_history_with_summary(
    messages: list[dict],
    keep_last_n: int = KEEP_LAST_N_MESSAGES,
    model: str | None = None,
) -> list[dict]:
    """Like compress_history but replaces dropped messages with an LLM-generated
    summary instead of a trim marker.

    More expensive (requires an LLM call) but produces a richer summary that
    the agent can use to understand what happened earlier in the task.

    Falls back to plain compression if the summary call fails.
    """
    if len(messages) <= keep_last_n + 1:
        return messages

    first = messages[0]
    to_summarise = messages[1:-keep_last_n]
    recent = messages[-keep_last_n:]

    summary_text = _summarise_messages(to_summarise, model)

    summary_message = {
        "role": "user",
        "content": f"[EARLIER CONTEXT SUMMARY]\n{summary_text}",
    }

    return [first, summary_message] + recent


def _summarise_messages(messages: list[dict], model: str | None = None) -> str:
    """Ask the LLM to summarise a slice of the message history."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    history_text = _messages_to_text(messages)

    summarise_prompt = (
        "The following is a partial history of an agent working on a coding task. "
        "Summarise what happened concisely: what files were created or modified, "
        "what commands were run, what succeeded, what failed, and any key decisions made. "
        "Be factual and brief — this summary will be injected back into the agent's context.\n\n"
        f"<history>\n{history_text}\n</history>"
    )

    try:
        if provider == "anthropic":
            _model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=_model,
                max_tokens=512,
                messages=[{"role": "user", "content": summarise_prompt}],
            )
            return response.content[0].text

        elif provider == "gemini":
            from google import genai as _genai
            _model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            client = _genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            response = client.models.generate_content(
                model=_model,
                contents=summarise_prompt,
            )
            return response.text

    except Exception as e:
        return f"[Summary unavailable: {e}]"

    return "[Summary unavailable]"


def _messages_to_text(messages: list[dict]) -> str:
    """Flatten messages to a readable text representation for summarisation."""
    lines = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]

        if isinstance(content, str):
            lines.append(f"{role}: {content}")

        elif isinstance(content, list):
            for block in content:
                btype = block.get("type", "") if isinstance(block, dict) else ""
                if btype == "text":
                    lines.append(f"{role}: {block['text']}")
                elif btype == "tool_use":
                    args = json.dumps(block.get("input", {}))[:200]
                    lines.append(f"{role} [tool_call]: {block['name']}({args})")
                elif btype == "tool_result":
                    result = str(block.get("content", ""))[:300]
                    lines.append(f"{role} [tool_result]: {result}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task-level summary (for session persistence)
# ---------------------------------------------------------------------------

def summarise_task(task_messages: list[dict], model: str | None = None) -> str:
    """Produce a summary of a completed task's full message history.

    Used when saving to session: the full task messages are replaced with
    this summary so the session file stays bounded over time.
    """
    return _summarise_messages(task_messages, model)
