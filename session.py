"""
session.py — cross-task conversation persistence.

A session is a JSON file that stores the conversation history across multiple
agent runs. Each task's messages are appended to the session, then compressed
to a summary after the task completes — keeping the file bounded over time.

File location: <workdir>/.session.json  (inside the sandbox, not source root)

Session file format:
{
    "session_id":   "sess_20260602_143022",
    "created_at":   "2026-06-02T14:30:22",
    "updated_at":   "2026-06-02T15:12:44",
    "workdir":      "./sandbox",
    "provider":     "anthropic",
    "model":        "claude-sonnet-4-5",
    "total_tasks":  3,
    "messages":     [ ... full unified message history ... ]
}
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from context import summarise_task

SESSION_FILENAME = ".session.json"


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class Session:
    session_id: str
    created_at: str
    updated_at: str
    workdir: str
    provider: str
    model: str
    total_tasks: int = 0
    messages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id":  self.session_id,
            "created_at":  self.created_at,
            "updated_at":  self.updated_at,
            "workdir":     self.workdir,
            "provider":    self.provider,
            "model":       self.model,
            "total_tasks": self.total_tasks,
            "messages":    self.messages,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(
            session_id=  data["session_id"],
            created_at=  data["created_at"],
            updated_at=  data["updated_at"],
            workdir=     data["workdir"],
            provider=    data["provider"],
            model=       data["model"],
            total_tasks= data.get("total_tasks", 0),
            messages=    data.get("messages", []),
        )


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def _session_path(workdir: str) -> Path:
    """Session file lives inside the sandbox so it's co-located with generated files."""
    return Path(workdir) / SESSION_FILENAME


def new_session(workdir: str, provider: str, model: str) -> Session:
    """Create a fresh session."""
    now = _now()
    session_id = f"sess_{now.replace(':', '').replace('-', '').replace('T', '_')[:15]}"
    return Session(
        session_id=session_id,
        created_at=now,
        updated_at=now,
        workdir=workdir,
        provider=provider,
        model=model,
    )


def load_session(workdir: str) -> Session | None:
    """Load a session from disk. Returns None if no session file exists."""
    path = _session_path(workdir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Session.from_dict(data)
    except Exception as e:
        print(f"  [session] failed to load {path}: {e}")
        return None


def save_session(session: Session) -> None:
    """Persist the session to disk."""
    path = _session_path(session.workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    session.updated_at = _now()
    path.write_text(
        json.dumps(session.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def delete_session(workdir: str) -> None:
    """Delete the session file (used when --new-session is passed)."""
    path = _session_path(workdir)
    if path.exists():
        path.unlink()
        print(f"  [session] deleted {path}")


# ---------------------------------------------------------------------------
# Task integration
# ---------------------------------------------------------------------------

def start_task(session: Session, task: str) -> list[dict]:
    """Begin a new task on an existing session.

    Returns the messages list to use for this task run — the session's
    existing history plus the new user task appended.
    """
    # New task message
    task_message = {"role": "user", "content": task}

    # The working messages = session history + this new task
    messages = session.messages + [task_message]
    return messages


def finish_task(
    session: Session,
    task_messages: list[dict],
    compress: bool = True,
) -> None:
    """Called when a task completes. Updates the session with the task history.

    If compress=True, the task's full messages are replaced with a summary
    to keep the session file bounded. The summary is generated via the LLM.

    If compress=False, all messages are appended verbatim (useful for
    debugging or short tasks where you want to keep the full history).
    """
    session.total_tasks += 1

    if compress and len(task_messages) > 3:
        # Summarise the task into a single message pair:
        #   user:      the original task
        #   assistant: the summary of what happened
        original_task = task_messages[0]["content"]
        summary = summarise_task(task_messages[1:])  # skip the user task message

        session.messages.append({
            "role": "user",
            "content": original_task,
        })
        session.messages.append({
            "role": "assistant",
            "content": (
                f"[TASK SUMMARY — task {session.total_tasks}]\n{summary}"
            ),
        })
    else:
        # Append verbatim
        session.messages.extend(task_messages)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_session_info(session: Session) -> None:
    print(f"  [session] id={session.session_id}  tasks={session.total_tasks}  "
          f"messages={len(session.messages)}  created={session.created_at[:10]}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
