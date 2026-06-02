# minimal-agent — Plan & Low-Level Design

## What this project is

The first working agent loop. No RAG, no AST, no hybrid search. Just the skeleton:
an LLM that can reason about a task, a set of tools it can call, and a loop that
executes those calls until the task is done.

This is Phase 1 of the autonomous coding agent journey. Everything built here
becomes the foundation that Phases 2–5 plug into.

---

## Goal

By the end of this project, you should be able to run:

```bash
python agent.py "create a Python function that reads a CSV and returns the row count, save it to utils.py"
```

And watch the agent:
1. Think about what it needs to do
2. Call `write_file` to create the function
3. Call `run_command` to verify it works
4. Observe the result, fix if needed
5. Declare done

A real agent completing a real task. Small, but real.

---

## What you will learn

| Concept | What it teaches |
|---------|----------------|
| Tool calling mechanics | LLM outputs a structured "I want to call X with args Y" — your code executes it |
| System prompt design | How agent prompts differ from chat prompts — tools, reasoning format, termination |
| ReAct pattern | Reasoning + Acting — think before every tool call, observe after |
| Message history | How the conversation array grows as the agent works — foundation for context management |
| Failure modes | Wrong args, infinite loops, early termination, hallucinated paths — and how to handle them |

---

## Project structure

```
minimal-agent/
  plan.md           ← this file
  agent.py          ← the loop — entry point
  llm.py            ← Anthropic API wrapper
  tools.py          ← tool schemas + implementations
  prompts.py        ← system prompt
  .env              ← ANTHROPIC_API_KEY (never commit)
  .env.example      ← template for the above
  pyproject.toml    ← dependencies
```

---

## Low-Level Design

### How the pieces connect

```
User
  │
  │  task string (CLI arg)
  ▼
agent.py  ──────────────────────────────────────────────────────┐
  │                                                              │
  │  1. Build initial messages array                            │
  │     [system_prompt, user_task]                              │
  │                                                             │
  │  2. Enter the loop                                          │
  │                                                             │
  │  ┌──────────────────────────────────────────────────────┐  │
  │  │                  AGENT LOOP                          │  │
  │  │                                                      │  │
  │  │   messages ──► llm.py ──► Anthropic API             │  │
  │  │                    │                                 │  │
  │  │                    ▼                                 │  │
  │  │             parse response                           │  │
  │  │                    │                                 │  │
  │  │         ┌──────────┴──────────┐                     │  │
  │  │         ▼                     ▼                     │  │
  │  │    tool_use block        text block                 │  │
  │  │         │                     │                     │  │
  │  │         ▼                     ▼                     │  │
  │  │   tools.py dispatch      print to user             │  │
  │  │   execute the tool       exit loop ✅              │  │
  │  │         │                                           │  │
  │  │         ▼                                           │  │
  │  │   append tool result                                │  │
  │  │   to messages                                       │  │
  │  │         │                                           │  │
  │  │         └──────────► loop again                     │  │
  │  │                                                      │  │
  │  └──────────────────────────────────────────────────────┘  │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

---

### Module: `prompts.py`

Holds the system prompt as a string constant. This is where the agent's
"personality" and rules live.

**What the system prompt must define:**
- What the agent is (a coding assistant that operates on files)
- What tools it has (name, what each does, when to use it)
- How to reason (think step by step before every action)
- How to terminate (when the task is done, say so in plain text — no tool call)
- Safety rules (don't delete files unless explicitly asked, confirm destructive ops)

**Structure of the system prompt:**

```
You are an autonomous coding agent. You help users complete programming tasks
by reading, writing, and running code.

You have access to the following tools:
  - read_file(path): Read the contents of a file
  - write_file(path, content): Write content to a file (creates if not exists)
  - run_command(command): Run a shell command, returns stdout, stderr, exit code

How to work:
  1. Think through what you need to do before calling any tool
  2. Call one tool at a time
  3. Observe the result before deciding the next step
  4. When the task is complete, respond with plain text — no tool call

Rules:
  - Never delete files unless explicitly asked
  - Always verify your work by running the code after writing it
  - If a command fails, read the error and fix it before giving up
```

---

### Module: `llm.py`

A thin wrapper around the Anthropic Python SDK. Responsible for:
- Holding the client and model config
- Accepting messages + tool schemas, returning the raw response
- Nothing else — no parsing, no business logic

**Key Anthropic API concepts:**

The Anthropic messages API works like this:

```python
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=4096,
    system=SYSTEM_PROMPT,           # system prompt is separate from messages
    tools=TOOL_SCHEMAS,             # list of tool definitions
    messages=conversation_history   # list of {role, content} dicts
)
```

The response `content` is a list of blocks. Each block is either:
- `TextBlock` — plain text response (agent is done or thinking out loud)
- `ToolUseBlock` — agent wants to call a tool

```python
# TextBlock
{"type": "text", "text": "I'll start by reading the file..."}

# ToolUseBlock
{"type": "tool_use", "id": "toolu_01...", "name": "read_file", "input": {"path": "src/main.py"}}
```

The `stop_reason` field tells you why the model stopped:
- `"tool_use"` — it wants to call a tool, keep looping
- `"end_turn"` — it's done, exit the loop

**Interface:**

```python
def call_llm(messages: list[dict], tools: list[dict]) -> anthropic.types.Message:
    """Send messages to Claude, return the raw response."""
```

---

### Module: `tools.py`

Two responsibilities in one file:
1. **Tool schemas** — the JSON definitions you pass to the LLM so it knows what tools exist
2. **Tool implementations** — the actual Python functions that run when the LLM calls a tool

**Tool schema format (Anthropic):**

```python
{
    "name": "read_file",
    "description": "Read the contents of a file at the given path. Use this to understand existing code before modifying it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The file path to read, relative to the current working directory"
            }
        },
        "required": ["path"]
    }
}
```

The `description` field is critical — the LLM reads this to decide when to use the tool.
A vague description leads to wrong tool selection.

**The three tools:**

`read_file(path: str) -> str`
- Opens the file, returns its content as a string
- If file not found, return a clear error string (don't raise — the agent should handle it)
- Safety: reject paths with `..` to prevent directory traversal

`write_file(path: str, content: str) -> str`
- Creates parent directories if they don't exist
- Writes content, returns a success/failure message
- Returns the number of lines written so the agent can confirm

`run_command(command: str) -> str`
- Runs via `subprocess.run`, captures stdout + stderr
- Returns a formatted string: `exit_code`, `stdout`, `stderr`
- Timeout: 30 seconds — kill if exceeded
- Safety: block obviously dangerous commands (`rm -rf /`, `sudo`, etc.)

**Tool dispatcher:**

```python
TOOL_IMPLEMENTATIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}

def execute_tool(name: str, args: dict) -> str:
    """Look up and call the right tool implementation."""
    fn = TOOL_IMPLEMENTATIONS.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'"
    return fn(**args)
```

---

### Module: `agent.py`

The entry point and the loop. This is the core of the project.

**Message history structure:**

The Anthropic API expects messages in this format:

```python
# User message
{"role": "user", "content": "Add input validation to login.py"}

# Assistant message with tool call
{"role": "assistant", "content": [
    {"type": "text", "text": "I'll read the file first."},
    {"type": "tool_use", "id": "toolu_01...", "name": "read_file", "input": {"path": "login.py"}}
]}

# Tool result (must follow the assistant message that called it)
{"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "toolu_01...", "content": "def login():\n    pass"}
]}
```

Note: tool results go back as `role: "user"` — this is Anthropic's convention.
The `tool_use_id` must match the `id` from the `ToolUseBlock`.

**The loop:**

```python
def run_agent(task: str):
    messages = [{"role": "user", "content": task}]

    while True:
        response = call_llm(messages, TOOL_SCHEMAS)

        # Append assistant's response to history
        messages.append({"role": "assistant", "content": response.content})

        # Check stop reason
        if response.stop_reason == "end_turn":
            # Agent is done — print final message and exit
            for block in response.content:
                if block.type == "text":
                    print(block.text)
            break

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[tool] {block.name}({block.input})")
                result = execute_tool(block.name, block.input)
                print(f"[result] {result[:200]}...")  # preview
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        # Feed results back
        messages.append({"role": "user", "content": tool_results})
```

**Safety: max iterations**

Add a hard cap — if the agent hasn't finished after N iterations, stop and report.
Prevents infinite loops during development.

```python
MAX_ITERATIONS = 20
iteration = 0

while iteration < MAX_ITERATIONS:
    iteration += 1
    ...
```

---

## Message flow — concrete example

Task: `"write a Python function that adds two numbers, save to math_utils.py, then verify it runs"`

```
Turn 1
  User:      "write a Python function..."
  Assistant: [TextBlock: "I'll write the function first."]
             [ToolUseBlock: write_file(path="math_utils.py", content="def add(a, b):\n    return a + b")]
  stop_reason: "tool_use"

Turn 2
  User:      [ToolResult: "Written 2 lines to math_utils.py"]
  Assistant: [TextBlock: "Now I'll verify it runs."]
             [ToolUseBlock: run_command(command="python -c 'from math_utils import add; print(add(2,3))'")]
  stop_reason: "tool_use"

Turn 3
  User:      [ToolResult: "exit_code: 0\nstdout: 5\nstderr: "]
  Assistant: [TextBlock: "The function works correctly. add(2, 3) returns 5. Task complete."]
  stop_reason: "end_turn"

→ Loop exits. Print final message.
```

---

## What to build first

1. `prompts.py` — write the system prompt. No code, just text. Get this right first.
2. `tools.py` — implement the three tools + schemas. Test each function manually before wiring to the agent.
3. `llm.py` — the API wrapper. Test with a simple "hello" message before adding tools.
4. `agent.py` — the loop. Wire everything together. Start with `MAX_ITERATIONS = 5` to avoid runaway costs while testing.

---

## Dependencies

```toml
[project]
name = "minimal-agent"
version = "0.1.0"
requires-python = ">=3.11"

[project.dependencies]
anthropic = ">=0.40.0"
python-dotenv = ">=1.0.0"
```

---

## Environment setup

```bash
# .env (never commit this)
ANTHROPIC_API_KEY=sk-ant-...
```

```bash
uv venv
uv sync
python agent.py "your task here"
```

---

## What comes next (Phase 2)

Once this loop works reliably, Phase 2 adds context management:
- Token counting (how full is the context window?)
- Sliding window compression (summarize old turns when window fills)
- Retrieval trigger (when to search the codebase vs rely on existing context)

The message history structure you build here is exactly what Phase 2 operates on.

→ See `learnings/00-architecture.md` for the full roadmap.
