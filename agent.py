"""
agent.py — the entry point and the loop.

Usage:
    python agent.py "your task here"

The loop:
    1. Build initial messages with the user's task
    2. Call the LLM (provider selected by LLM_PROVIDER in .env)
    3. If stop_reason == "end_turn"  → print final message, exit
    4. If stop_reason == "tool_use"  → execute each tool call, feed results back, loop
    5. Hard cap at MAX_ITERATIONS to prevent runaway loops
"""

import os
import sys
from dotenv import load_dotenv

from llm import call_llm, build_assistant_message, build_tool_result_message
from tools import TOOL_SCHEMAS, execute_tool

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 20  # hard cap — prevents infinite loops during development


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_separator(label: str = "") -> None:
    width = 60
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * (pad)}")
    else:
        print("─" * width)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(task: str) -> None:
    """Run the agent loop for the given task."""
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "anthropic")
    print_separator("task")
    print(task)
    print(f"\nprovider: {provider}")
    print_separator()

    # Seed the conversation with the user's task
    messages: list[dict] = [
        {"role": "user", "content": task}
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n[iteration {iteration}]")

        # ── Call the LLM ──────────────────────────────────────────────────
        response = call_llm(messages, TOOL_SCHEMAS)
        if iteration == 1 or response.model_used:
            print(f"  model: {response.model_used}")

        # ── Append assistant response to history ──────────────────────────
        messages.append(build_assistant_message(response))

        # ── Check stop reason ─────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            print_separator("done")
            for block in response.content:
                if block.type == "text":
                    print(block.text)
            print_separator()
            return

        # ── Process tool calls ────────────────────────────────────────────
        if response.stop_reason == "tool_use":
            for block in response.content:
                # Print any thinking text before the tool call
                if block.type == "text" and block.text.strip():
                    print(f"  think: {block.text.strip()}")

                elif block.type == "tool_use":
                    # Show what the agent is doing
                    args_preview = ", ".join(
                        f"{k}={repr(v)[:60]}" for k, v in block.input.items()
                    )
                    print(f"  call:  {block.name}({args_preview})")

                    # Execute the tool
                    result = execute_tool(block.name, block.input)

                    # Preview the result (first 200 chars)
                    preview = result[:200].replace("\n", " ")
                    if len(result) > 200:
                        preview += "..."
                    print(f"  result: {preview}")

                    # Feed result back — one message per tool call
                    messages.append(
                        build_tool_result_message(block.id, block.name, result)
                    )

        else:
            print(f"[unexpected stop_reason: {response.stop_reason}]")
            break

    # Reached iteration cap
    print_separator("stopped")
    print(f"Reached maximum iterations ({MAX_ITERATIONS}). Task may be incomplete.")
    print_separator()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py \"your task here\"")
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    run_agent(task)
