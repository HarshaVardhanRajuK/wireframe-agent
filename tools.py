"""
Tool definitions and implementations for the minimal agent.

Each tool has two parts:
  1. A schema — the JSON definition passed to the LLM so it knows the tool exists
  2. An implementation — the Python function that runs when the LLM calls it

All file operations (read_file, write_file) are scoped to a sandbox directory
set via set_workdir(). This keeps agent-generated files separate from source code.
run_command executes with the sandbox as its working directory.
"""

import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Sandbox directory
# ---------------------------------------------------------------------------

_WORKDIR: Path = Path("./sandbox")


def set_workdir(path: str) -> None:
    """Set the sandbox directory for all file operations. Creates it if needed."""
    global _WORKDIR
    _WORKDIR = Path(path).resolve()
    _WORKDIR.mkdir(parents=True, exist_ok=True)
    print(f"  [sandbox: {_WORKDIR}]")


def _resolve(path: str) -> Path:
    """Resolve a relative path against the sandbox. Blocks directory traversal."""
    if ".." in Path(path).parts:
        raise ValueError(f"Path traversal not allowed: {path}")
    resolved = (_WORKDIR / path).resolve()
    # Ensure the resolved path is still inside the sandbox
    resolved.relative_to(_WORKDIR)  # raises ValueError if outside
    return resolved


# ---------------------------------------------------------------------------
# Safety config
# ---------------------------------------------------------------------------

BLOCKED_COMMAND_PREFIXES = [
    "rm -rf /",
    "sudo rm",
    "mkfs",
    "dd if=",
    "> /dev/",
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    """Read the contents of a file and return it as a string."""
    try:
        file_path = _resolve(path)
    except ValueError as e:
        return f"Error: {e}"

    if not file_path.exists():
        return f"Error: file not found: {path}"
    if not file_path.is_file():
        return f"Error: path is not a file: {path}"

    try:
        content = file_path.read_text(encoding="utf-8")
        line_count = content.count("\n") + 1
        return f"[{line_count} lines]\n{content}"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    try:
        file_path = _resolve(path)
    except ValueError as e:
        return f"Error: {e}"

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        line_count = content.count("\n") + 1
        return f"Written {line_count} lines to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def run_command(command: str) -> str:
    """Run a shell command in the sandbox directory."""
    for prefix in BLOCKED_COMMAND_PREFIXES:
        if command.strip().startswith(prefix):
            return f"Error: command blocked for safety: '{prefix}...'"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(_WORKDIR),  # run inside the sandbox
        )
        parts = [f"exit_code: {result.returncode}"]
        if result.stdout.strip():
            parts.append(f"stdout:\n{result.stdout.rstrip()}")
        if result.stderr.strip():
            parts.append(f"stderr:\n{result.stderr.rstrip()}")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error running command: {e}"


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

TOOL_IMPLEMENTATIONS: dict = {
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}


def execute_tool(name: str, args: dict) -> str:
    """Look up and call the right tool implementation. Returns a string result."""
    fn = TOOL_IMPLEMENTATIONS.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'"
    try:
        return fn(**args)
    except TypeError as e:
        return f"Error: wrong arguments for tool '{name}': {e}"


# ---------------------------------------------------------------------------
# Tool schemas (passed to the LLM)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file at the given path. "
            "Paths are relative to the sandbox working directory. "
            "Use this to understand existing code before modifying it. "
            "Always read a file before writing to it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read, relative to the current working directory.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file. Paths are relative to the sandbox working directory. "
            "Creates the file and any parent directories if they don't exist. "
            "Overwrites the file if it already exists. "
            "Use this to create new files or update existing ones."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write to, relative to the current working directory.",
                },
                "content": {
                    "type": "string",
                    "description": "The full content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command and return its stdout, stderr, and exit code. "
            "Use this to run Python scripts, execute tests, verify code works, "
            "or inspect the environment. Commands time out after 30 seconds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run.",
                }
            },
            "required": ["command"],
        },
    },
]
