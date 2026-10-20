"""DSH-compatible coding agent: the same three tools the evaluation harness serves.

Tool names, parameters and defaults mirror the DeepSeek Harness (DSH) evaluation
profile, so a trajectory produced here can be consumed by the same student that is
scored by that harness. There is deliberately no ``finish`` and no ``think`` tool:
DSH terminates on natural stop, and an extra completion action would both change
the measured behaviour and teach the student an action that does not exist at
evaluation time.
"""
from __future__ import annotations

import posixpath
from typing import Any

from aweagent.core.runtime.protocol import RuntimeSession
from aweagent.core.tool.protocol import Tool

READ_DEFAULT_LIMIT = 400
READ_MAX_LIMIT = 2000
BASH_TIMEOUT_S = 120
BASH_MAX_TIMEOUT_S = 600


def _confine(path: str, workdir: str) -> str:
    """Confine file tools to the session workspace.

    Batch runs create one agent before the per-instance workdir is known, so the
    guard accepts any absolute path under the Scale-SWE workspace root
    (``/workspace``) and rejects traversal. When a concrete workdir is supplied
    (single-instance runs) the tighter prefix check applies.
    """
    if not path:
        raise ValueError("file_path must be a non-empty string")
    if ".." in path.split("/"):
        raise ValueError(f"path traversal is not allowed: {path}")
    root = workdir.rstrip("/") or "/workspace"
    absolute = path if path.startswith("/") else posixpath.join(root, path)
    normalized = posixpath.normpath(absolute)
    if normalized == root or normalized.startswith(root + "/"):
        return normalized
    if root == "/workspace" and normalized.startswith("/workspace/"):
        return normalized
    raise ValueError(f"path is outside the session workspace: {path}")


class DshBashTool(Tool):
    """``bash``: command plus a required one-line description."""

    def __init__(self, timeout: int = BASH_TIMEOUT_S, max_timeout: int = BASH_MAX_TIMEOUT_S):
        self._timeout = timeout
        self._max_timeout = max_timeout

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return ("Execute a bash command (`bash -c`) and return its stdout/stderr. "
                "Commands run inside the task workspace. Long-running commands must finish "
                "within the timeout.")

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {
            "command": {"type": "string", "description": "Exact bash command to execute."},
            "description": {"type": "string",
                            "description": "5-10 word description of what the command does."},
        }, "required": ["command", "description"]}

    async def execute(self, params: dict[str, Any], session: RuntimeSession | None = None) -> str:
        if session is None:
            return "Error: bash requires a runtime session."
        command = str(params.get("command") or "")
        if not command.strip():
            return "Error: invalid command: expected a non-empty string"
        if not str(params.get("description") or "").strip():
            return "Error: invalid description: expected a non-empty string"
        timeout = params.get("timeout", self._timeout)
        try:
            timeout = min(int(timeout), self._max_timeout)
        except (TypeError, ValueError):
            timeout = self._timeout
        result = await session.execute(command, timeout=timeout)
        body = (getattr(result, "output", "") or "")
        exit_code = getattr(result, "exit_code", None)
        if exit_code not in (None, 0):
            body = f"{body}\n[exit code: {exit_code}]"
        return body or "[no output]"


class DshReadTool(Tool):
    """``read``: line-numbered view with offset/limit, mirroring DSH defaults."""

    def __init__(self, workdir: str):
        self._workdir = workdir

    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        return ("Read a UTF-8 text file and return line-numbered content. Use offset/limit to "
                "page through large files.")

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "Path to the file to read."},
            "offset": {"type": "integer", "description": "1-based first line. Defaults to 1."},
            "limit": {"type": "integer",
                      "description": f"Max lines to return (default {READ_DEFAULT_LIMIT}, "
                                     f"max {READ_MAX_LIMIT})."},
        }, "required": ["file_path"]}

    async def execute(self, params: dict[str, Any], session: RuntimeSession | None = None) -> str:
        if session is None:
            return "Error: read requires a runtime session."
        try:
            path = _confine(str(params.get("file_path") or ""), self._workdir)
            offset = max(1, int(params.get("offset") or 1))
            limit = int(params.get("limit") or READ_DEFAULT_LIMIT)
            if limit > READ_MAX_LIMIT:
                return f"Error: limit must be less than or equal to {READ_MAX_LIMIT}"
            data = await session.download_file(path)
        except Exception as error:  # noqa: BLE001 - surfaced to the model as text
            return f"Error: {error}"
        lines = data.decode("utf-8", errors="replace").split("\n")
        window = lines[offset - 1: offset - 1 + max(1, limit)]
        body = "\n".join(f"{offset + index:6d}\t{text}" for index, text in enumerate(window))
        return f"{body}\n\n[{len(lines)} lines total]"


class DshEditTool(Tool):
    """``edit``: literal replacement, ambiguous matches rejected unless replace_all."""

    def __init__(self, workdir: str):
        self._workdir = workdir

    @property
    def name(self) -> str:
        return "edit"

    @property
    def description(self) -> str:
        return ("Replace a literal string in an existing UTF-8 text file. By default the old "
                "string must appear exactly once; otherwise pass replace_all.")

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "Path to the file to edit."},
            "old_string": {"type": "string", "description": "Literal text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean",
                            "description": "Replace every match. Defaults to false."},
        }, "required": ["file_path", "old_string", "new_string"]}

    async def execute(self, params: dict[str, Any], session: RuntimeSession | None = None) -> str:
        if session is None:
            return "Error: edit requires a runtime session."
        old = str(params.get("old_string") or "")
        new = str(params.get("new_string") or "")
        replace_all = bool(params.get("replace_all") or False)
        if not old:
            return "Error: old_string must not be empty"
        try:
            path = _confine(str(params.get("file_path") or ""), self._workdir)
            data = await session.download_file(path)
        except Exception as error:  # noqa: BLE001
            return f"Error: {error}"
        text = data.decode("utf-8")
        matches = text.count(old)
        if matches == 0:
            return "Error: old_string not found"
        if matches > 1 and not replace_all:
            return (f"Error: old_string matched {matches} times in {path!r}; provide a more "
                    "specific old_string or set replace_all to true")
        updated = text.replace(old, new, -1 if replace_all else 1)
        await session.upload_file(path, updated.encode("utf-8"))
        count = matches if replace_all else 1
        return f"edited {path}: {count} replacement(s)"


def build_tools(workdir: str, *, bash_timeout: int = BASH_TIMEOUT_S,
                bash_max_timeout: int = BASH_MAX_TIMEOUT_S) -> list[Tool]:
    return [DshBashTool(bash_timeout, bash_max_timeout), DshReadTool(workdir), DshEditTool(workdir)]
