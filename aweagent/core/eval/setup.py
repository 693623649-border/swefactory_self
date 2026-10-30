"""PreAgentSetup — prepare the container environment before agent execution.

Encapsulates:

- Running instance setup commands (from ``pre_commands`` / ``setup_commands``)
- Removing future git commits (prevent data leakage)

Usage::

    setup = PreAgentSetup(session, instance.workdir)
    await setup.prepare(instance)
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
import shlex
from typing import TYPE_CHECKING

from aweagent.core.task.types import Instance

if TYPE_CHECKING:
    from aweagent.core.runtime.protocol import RuntimeSession

logger = logging.getLogger(__name__)

# Shell snippet that commits all current changes as a "pre-agent" snapshot.
_GIT_COMMIT_PRE_AGENT = (
    'git submodule update --recursive --no-fetch && '
    'git config user.email "pre-agent@awe-agent.local" && '
    'git config user.name "Pre-Agent" && '
    'git add -A && '
    'git commit -m "pre-agent commit"'
)

def hardening_command(workdir: str, *, audit_copies: bool = False) -> str:
    source = Path(__file__).with_name('workspace_hardening.py').read_text(encoding='utf-8')
    return ('python -c ' + shlex.quote(source) + ' --root ' + shlex.quote(workdir)
            + ' --sanitize-git' + (' --audit-copies' if audit_copies else ''))


class PreAgentSetup:
    """Prepare the container environment before agent execution or evaluation.

    Encapsulates:

    - Running instance setup commands (from ``pre_commands``)
    - Removing future git commits (prevent data leakage)
    """

    def __init__(self, session: RuntimeSession, workdir: str) -> None:
        self.session = session
        self.workdir = workdir
        self.hardening_receipt = None

    async def run_setup_commands(self, commands: list[str]) -> None:
        """Execute setup commands (from ``instance.setup_commands``)."""
        for cmd in commands:
            result = await self.session.execute(cmd, cwd=self.workdir, timeout=300)
            if not result.success:
                logger.warning(
                    "Setup command failed (exit %d): %s -> %s",
                    result.exit_code,
                    cmd[:120],
                    result.stderr[:200],
                )

    async def commit_and_get_id(self) -> str | None:
        """Commit current state, return HEAD SHA."""
        await self.session.execute(
            _GIT_COMMIT_PRE_AGENT, cwd=self.workdir, timeout=120,
        )
        result = await self.session.execute(
            "git rev-parse HEAD", cwd=self.workdir, timeout=120,
        )
        if result.success and result.stdout.strip():
            return result.stdout.strip()
        return None

    async def remove_future_commits(self) -> None:
        """Remove future refs/objects recursively, and require a successful audit."""
        result = await self.session.execute(hardening_command(self.workdir),
                                            cwd=self.workdir, timeout=300)
        if not result.success:
            raise RuntimeError('pre-solver Git hardening failed: ' + result.output[-1500:])
        receipt = json.loads(result.stdout)
        if receipt.get('git_passed') is not True:
            raise RuntimeError('pre-solver Git hardening did not pass')
        self.hardening_receipt = receipt

    async def prepare(self, instance: Instance) -> str | None:
        """Full pre-agent preparation: setup → commit snapshot → remove future commits.

        Returns the pre-agent commit SHA (or ``None`` on failure).
        """
        await self.run_setup_commands(instance.setup_commands)
        commit_id = await self.commit_and_get_id()
        await self.remove_future_commits()
        return commit_id
