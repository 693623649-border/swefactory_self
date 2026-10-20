"""DSH-compatible scaffold: three tools, native tool calling, natural stop."""
from __future__ import annotations

from typing import Any

from aweagent.core.llm.format import get_tool_format
from aweagent.core.tool.protocol import Tool
from aweagent.scaffold.dsh_compat.contract_review import CONTRACT_REVIEW
from aweagent.scaffold.dsh_compat.tools import BASH_TIMEOUT_S, BASH_MAX_TIMEOUT_S, build_tools
from aweagent.scaffold.search_swe import SearchSWEAgent

# Verbatim from eval/profiles/swe-agent-v3s/cordis.patch.yml: the persona the
# student is scored with. Keep the two strings identical.
DSH_PERSONA = (
    "You are a software engineer fixing one repository task. Read, edit, and "
    "test only inside the session workspace. The workspace is the same tree as "
    "/testbed inside the task container; both the host path and /testbed are "
    "valid for every tool, and every bash command runs in that same tree. "
    "Do not use the web, other models, or subagents. You must apply the change "
    "to the files: a final answer that only describes the fix is a failure, and "
    "the uncommitted diff is the submitted patch. Do not stop until you have "
    "edited the files and run the target tests; do not claim the task is solved "
    "unless the tests you actually ran passed."
)

# Data-side constraints on top of the scoring persona: they prevent the exact
# contamination the acceptance gates reject, and are not part of the eval prompt.
DATA_CONTRACT = (
    "\n\nData contract: do not modify test files, test fixtures, or snapshot files; "
    "the grader restores them before scoring, so editing them cannot help. Run the "
    "target tests after editing and only then stop."
)


class DSHCompatAgent(SearchSWEAgent):
    """SearchSWE with the DSH tool surface, DSH persona, and no finish tool."""

    def __init__(self, workdir: str, *, bash_timeout: int = BASH_TIMEOUT_S,
                 bash_max_timeout: int = BASH_MAX_TIMEOUT_S,
                 contract_review: bool = False, **kwargs: Any):
        kwargs.setdefault("enable_search", False)
        kwargs.setdefault("enable_think", False)
        kwargs.setdefault("tool_call_format", "openai_function")
        super().__init__(**kwargs)
        self._workdir = workdir
        self._contract_review = contract_review
        self._tools = build_tools(workdir, bash_timeout=bash_timeout,
                                  bash_max_timeout=bash_max_timeout)
        self._format = get_tool_format("openai_function")

    def get_tools(self) -> list[Tool]:
        return list(self._tools)

    def get_system_prompt(self, task_info: dict[str, Any]) -> str:
        return DSH_PERSONA + DATA_CONTRACT + (CONTRACT_REVIEW if self._contract_review else "")

    @classmethod
    def from_config(cls, config: Any) -> "DSHCompatAgent":
        workdir = (getattr(config.runtime, "workdir", "") or "/workspace").rstrip("/") or "/workspace"
        return cls(workdir, bash_timeout=config.agent.bash_timeout,
                   bash_max_timeout=config.agent.bash_max_timeout,
                   contract_review=bool(config.extra.get("dsh_contract_review", False)))

    @classmethod
    def from_config_with_constraints(cls, config: Any, instance_constraints: Any) -> "DSHCompatAgent":
        # This scaffold has no search tools, so per-instance constraints are unused.
        return cls.from_config(config)

    def get_no_tool_call_prompt(self) -> str | None:
        """DSH v3s has no steer arm: stay silent so the comparison stays honest."""
        return None
