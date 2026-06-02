"""
LLM interface — supports two providers:
  - Gemini (default) via the google-genai SDK
  - Anthropic (Claude) via the anthropic SDK

Provider is selected by LLM_PROVIDER in .env:
  LLM_PROVIDER=gemini     → uses GEMINI_API_KEY  + GEMINI_MODEL
  LLM_PROVIDER=anthropic  → uses ANTHROPIC_API_KEY + ANTHROPIC_MODEL

Both providers return a shared LLMResponse so agent.py stays provider-agnostic.

Response anatomy:
    response.stop_reason  — "tool_use" (keep looping) or "end_turn" (done)
    response.content      — list of TextBlock | ToolUseBlock
    response.model_used   — which model actually responded
"""

import json
import os
from dataclasses import dataclass, field
from typing import Literal

import anthropic
from google import genai
from google.genai import types as gtypes

from prompts import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"

# ---------------------------------------------------------------------------
# Shared response types
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ToolUseBlock:
    type: Literal["tool_use"] = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    stop_reason: Literal["tool_use", "end_turn"]
    content: list[TextBlock | ToolUseBlock]
    model_used: str = ""


# ---------------------------------------------------------------------------
# Provider: Gemini
# ---------------------------------------------------------------------------

def _anthropic_tools_to_gemini(tools: list[dict]) -> list[gtypes.Tool]:
    """Convert our tool schema format to Gemini function declarations.

    Our schema (same as Anthropic):
        {"name": "...", "description": "...", "input_schema": {json-schema}}

    Gemini function_declarations:
        {"name": "...", "description": "...", "parameters": {json-schema}}
    """
    declarations = [
        gtypes.FunctionDeclaration(
            name=t["name"],
            description=t.get("description", ""),
            parameters=t.get("input_schema"),
        )
        for t in tools
    ]
    return [gtypes.Tool(function_declarations=declarations)]


def _call_gemini(
    messages: list[dict],
    tools: list[dict],
    model: str,
) -> LLMResponse:
    """Call Gemini via the google-genai SDK and normalise the response."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Convert our message history to Gemini's Content format.
    # Our internal format uses OpenAI-style roles + content.
    # Gemini uses role="user"/"model" and parts=[].
    gemini_history: list[gtypes.Content] = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        content = msg["content"]

        parts: list[gtypes.Part] = []

        if isinstance(content, str):
            parts.append(gtypes.Part(text=content))

        elif isinstance(content, list):
            for block in content:
                btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)

                if btype == "text":
                    text = block.get("text") if isinstance(block, dict) else block.text
                    if text:
                        parts.append(gtypes.Part(text=text))

                elif btype == "tool_use":
                    # Assistant called a tool — emit a function_call Part
                    b_name = block.get("name") if isinstance(block, dict) else block.name
                    b_input = block.get("input") if isinstance(block, dict) else block.input
                    b_id = block.get("id") if isinstance(block, dict) else block.id
                    parts.append(gtypes.Part(
                        function_call=gtypes.FunctionCall(name=b_name, args=b_input, id=b_id)
                    ))

                elif btype == "tool_result":
                    # Tool result — emit a function_response Part
                    result_content = block.get("content") if isinstance(block, dict) else block.content
                    tool_id = block.get("tool_use_id") if isinstance(block, dict) else block.tool_use_id
                    parts.append(gtypes.Part(
                        function_response=gtypes.FunctionResponse(
                            name=tool_id,
                            response={"result": result_content},
                        )
                    ))

        if parts:
            gemini_history.append(gtypes.Content(role=role, parts=parts))

    config = gtypes.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=_anthropic_tools_to_gemini(tools),
    )

    raw = client.models.generate_content(
        model=model,
        contents=gemini_history,
        config=config,
    )

    candidate = raw.candidates[0]
    content_out: list[TextBlock | ToolUseBlock] = []
    has_tool_call = False

    for part in candidate.content.parts:
        if part.text:
            content_out.append(TextBlock(text=part.text))
        elif part.function_call:
            has_tool_call = True
            fc = part.function_call
            content_out.append(ToolUseBlock(
                id=getattr(fc, "id", fc.name),  # Gemini 2.0+ includes id
                name=fc.name,
                input=dict(fc.args) if fc.args else {},
            ))

    stop = "tool_use" if has_tool_call else "end_turn"
    return LLMResponse(stop_reason=stop, content=content_out, model_used=model)


# ---------------------------------------------------------------------------
# Provider: Anthropic
# ---------------------------------------------------------------------------

def _call_anthropic(
    messages: list[dict],
    tools: list[dict],
    model: str,
) -> LLMResponse:
    """Call Claude via the Anthropic SDK and normalise the response."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    # Anthropic's tool_result block does not accept extra fields like tool_name.
    # Strip it out before sending — it's only needed by Gemini's FunctionResponse.
    clean_messages = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            clean_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    block = {k: v for k, v in block.items() if k != "tool_name"}
                clean_content.append(block)
            clean_messages.append({"role": msg["role"], "content": clean_content})
        else:
            clean_messages.append(msg)

    raw = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=clean_messages,
    )

    stop = "tool_use" if raw.stop_reason == "tool_use" else "end_turn"

    content: list[TextBlock | ToolUseBlock] = []
    for block in raw.content:
        if block.type == "text":
            content.append(TextBlock(text=block.text))
        elif block.type == "tool_use":
            content.append(ToolUseBlock(
                id=block.id,
                name=block.name,
                input=block.input,
            ))

    return LLMResponse(stop_reason=stop, content=content, model_used=model)


# ---------------------------------------------------------------------------
# Message history helpers
# ---------------------------------------------------------------------------

def build_assistant_message(response: LLMResponse) -> dict:
    """Build the assistant message to append to history after an LLM call.

    We store history in a unified format (list of typed block dicts) that
    both providers can reconstruct from.
    """
    blocks = []
    for block in response.content:
        if block.type == "text":
            blocks.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            blocks.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
    return {"role": "assistant", "content": blocks}


def build_tool_result_message(tool_use_id: str, tool_name: str, result: str) -> dict:
    """Build the tool result message.

    Both Gemini and Anthropic providers reconstruct the correct native format
    from this unified dict when building the request.

    Note: tool_name is needed by Gemini's FunctionResponse.
    """
    return {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "content": result,
        }],
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def call_llm(messages: list[dict], tools: list[dict]) -> LLMResponse:
    """Call the configured provider and return a normalised LLMResponse.

    Reads from environment:
        LLM_PROVIDER     — "gemini" (default) or "anthropic"
        GEMINI_MODEL     — override default Gemini model
        ANTHROPIC_MODEL  — override default Anthropic model
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        return _call_gemini(messages, tools, model)

    elif provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        return _call_anthropic(messages, tools, model)

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Must be 'gemini' or 'anthropic'."
        )
