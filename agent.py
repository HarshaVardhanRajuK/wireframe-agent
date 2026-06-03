"""
agent.py — the entry point and the loop.

Usage:
    # Single task, no session
    python agent.py "create a function that reverses a string"

    # Continue previous session (loads .session.json from workdir)
    python agent.py --continue "add a count_vowels function to utils.py"

    # Start fresh, ignoring any existing session
    python agent.py --new-session "start a new project"

    # Custom sandbox directory
    python agent.py "task" --workdir ./my-project

Context management:
    - Token usage is estimated after each iteration and printed
    - If pressure > 70%, history is compressed (keeps first + last 6 messages)
    - With --continue, session history is loaded so the agent remembers past tasks
    - Session is saved after each task completes (compressed to a summary)
"""

import argparse
import os
from dotenv import load_dotenv

from context import (
    TokenBudget,
    estimate_tokens,
    compress_history,
    COMPRESSION_THRESHOLD,
    KEEP_LAST_N_MESSAGES,
)
from llm import call_llm, build_assistant_message, build_tool_result_message
from session import (
    Session,
    new_session,
    load_session,
    save_session,
    delete_session,
    start_task,
    finish_task,
    print_session_info,
)
from tools import TOOL_SCHEMAS, execute_tool, set_workdir

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_separator(label: str = "") -> None:
    width = 60
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * pad}")
    else:
        print("─" * width)


def _get_model() -> str:
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(
    task: str,
    workdir: str = "./sandbox",
    use_session: bool = False,
    new_session_flag: bool = False,
) -> None:
    """Run the agent loop for the given task.

    Args:
        task:             The task string from the user.
        workdir:          Sandbox directory for all file operations.
        use_session:      If True, load and continue the existing session.
        new_session_flag: If True, delete the existing session and start fresh.
    """
    load_dotenv()
    set_workdir(workdir)

    provider = os.getenv("LLM_PROVIDER", "anthropic")
    model = _get_model()

    # ── Session setup ─────────────────────────────────────────────────────
    session: Session | None = None

    if new_session_flag:
        delete_session(workdir)
        session = new_session(workdir, provider, model)
        print(f"  [session] started new session {session.session_id}")

    elif use_session:
        session = load_session(workdir)
        if session:
            print_session_info(session)
        else:
            print(f"  [session] no existing session found, starting fresh")
            session = new_session(workdir, provider, model)

    else:
        # No session flag — stateless run (original behaviour)
        session = None

    # ── Build initial messages ────────────────────────────────────────────
    if session is not None:
        messages = start_task(session, task)
    else:
        messages = [{"role": "user", "content": task}]

    # ── Print task header ─────────────────────────────────────────────────
    print_separator("task")
    print(task)
    print(f"\nprovider: {provider}  |  model: {model}  |  workdir: {workdir}")
    if session:
        print(f"session:  {session.session_id}  (task #{session.total_tasks + 1})")
    print_separator()

    # ── Track where the task's own messages start ─────────────────────────
    # We need to know which messages belong to THIS task (not prior session history)
    # so we can compress/summarise just this task at the end.
    task_start_index = len(messages) - 1  # index of the new user task message

    # ── Token budget ──────────────────────────────────────────────────────
    budget = TokenBudget(model=model)

    # ── The loop ──────────────────────────────────────────────────────────
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n[iteration {iteration}]")

        # Token pressure check — compress if needed
        budget.used = estimate_tokens(messages)
        print(f"  {budget}")

        if budget.needs_compression:
            print(f"  [context] pressure {budget.pressure:.0%} — compressing history")
            # Compress only the task-specific messages, not prior session summaries
            session_prefix = messages[:task_start_index]
            task_messages  = messages[task_start_index:]
            task_messages  = compress_history(task_messages, KEEP_LAST_N_MESSAGES)
            messages = session_prefix + task_messages
            budget.used = estimate_tokens(messages)
            print(f"  [context] after compression: {budget}")

        # ── Call the LLM ──────────────────────────────────────────────────
        response = call_llm(messages, TOOL_SCHEMAS)
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

            # Save session if active
            if session is not None:
                task_messages = messages[task_start_index:]
                finish_task(session, task_messages, compress=True)
                save_session(session)
                print(f"  [session] saved → {session.workdir}/.session.json "
                      f"(task #{session.total_tasks})")
            return

        # ── Process tool calls ────────────────────────────────────────────
        if response.stop_reason == "tool_use":
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"  think: {block.text.strip()}")

                elif block.type == "tool_use":
                    args_preview = ", ".join(
                        f"{k}={repr(v)[:60]}" for k, v in block.input.items()
                    )
                    print(f"  call:  {block.name}({args_preview})")

                    result = execute_tool(block.name, block.input)

                    preview = result[:200].replace("\n", " ")
                    if len(result) > 200:
                        preview += "..."
                    print(f"  result: {preview}")

                    messages.append(
                        build_tool_result_message(block.id, block.name, result)
                    )

        else:
            print(f"[unexpected stop_reason: {response.stop_reason}]")
            break

    # ── Iteration cap reached ─────────────────────────────────────────────
    print_separator("stopped")
    print(f"Reached maximum iterations ({MAX_ITERATIONS}). Task may be incomplete.")
    if session is not None:
        task_messages = messages[task_start_index:]
        finish_task(session, task_messages, compress=True)
        save_session(session)
        print(f"  [session] saved (incomplete task #{session.total_tasks})")
    print_separator()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimal autonomous coding agent with context management"
    )
    parser.add_argument("task", nargs="+", help="The task for the agent to complete")
    parser.add_argument(
        "--workdir",
        default="./sandbox",
        help="Sandbox directory for file operations (default: ./sandbox)",
    )

    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument(
        "--continue",
        dest="use_session",
        action="store_true",
        help="Load and continue the existing session from workdir",
    )
    session_group.add_argument(
        "--new-session",
        dest="new_session",
        action="store_true",
        help="Delete any existing session and start fresh",
    )

    args = parser.parse_args()
    run_agent(
        task=" ".join(args.task),
        workdir=args.workdir,
        use_session=args.use_session,
        new_session_flag=args.new_session,
    )
