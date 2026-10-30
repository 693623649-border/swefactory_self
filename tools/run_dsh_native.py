"""Factory TaskRunner entry using the local_train eval DSH solver.

Input is an already exported public task fixture, never the hidden task record.
This opt-in smoke entry leaves the historical Python dsh_compat path unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

FACTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FACTORY))
from aweagent.core.agent.loop import AgentResult
from aweagent.core.agent.protocol import Agent
from aweagent.core.agent.trajectory import Action, Trajectory
from aweagent.core.llm.config import LLMConfig
from aweagent.core.llm.types import Message
from aweagent.core.runtime.config import RuntimeConfig
from aweagent.core.task.runner import TaskRunner
from aweagent.tasks.scale_swe.task import ScaleSWETask


class NativeTask(ScaleSWETask):
    """DSH driver owns the solve environment; factory still owns orchestration."""

    def requires_runtime(self) -> bool:
        return False

    def default_evaluator(self, timeout=None):
        return None


class NativeDshAgent(Agent):
    def __init__(self, driver, args):
        self.driver, self.args = driver, args

    async def step(self, context):
        raise RuntimeError("NativeDshAgent runs the actual DSH loop, not Python AgentLoop.step")

    def get_system_prompt(self, task_info):
        return ""

    def get_tools(self):
        return []

    def create_loop(self, context):
        return self

    async def run(self, prompt):
        args = self.args
        result = await asyncio.to_thread(
            self.driver.solve, args.task_file, args.instance_id, args.run_dir,
            "production", args.dsh_repo,
        )
        if result["terminal_status"] == "infrastructure_failure":
            raise RuntimeError("DSH infrastructure failure; inspect isolated run receipts")
        messages = json.loads(Path(result["messages_path"]).read_text(encoding="utf-8"))
        requests = json.loads((args.run_dir / "requests-and-assistants.json").read_text(encoding="utf-8"))
        trajectory = Trajectory()
        for index, item in enumerate(requests):
            assistant = item["assistant"]
            next_messages = requests[index + 1]["request"]["messages"] if index + 1 < len(requests) else []
            call_ids = {call["id"] for call in assistant.get("tool_calls", [])}
            observations = [m.get("content") or "" for m in next_messages
                            if m.get("role") == "tool" and m.get("tool_call_id") in call_ids]
            trajectory.add_step(index, Action(
                type="tool_call" if call_ids else "message", content=assistant.get("content"),
                tool_calls=assistant.get("tool_calls", []), reasoning_raw=assistant.get("reasoning_raw")),
                observations=observations)
        return AgentResult(
            trajectory=trajectory, patch=result.get("model_patch", ""),
            messages=[Message.from_dict(message) for message in messages],
            finish_reason=result["terminal_status"],
            metadata={"training_eligible": False, "native_receipts": str(args.run_dir)},
        )


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--dsh-repo", type=Path, required=True)
    args = parser.parse_args()
    for key in ("task_file", "run_dir", "eval_root", "dsh_repo"):
        setattr(args, key, getattr(args, key).resolve())
    sys.path.insert(0, str(args.eval_root / "swe"))
    import native_task

    public = native_task.load_task_row(args.instance_id, args.task_file)
    task = NativeTask(instances=[{
        "instance_id": public["instance_id"], "user": public["repo"].split("/")[0],
        "repo": public["repo"].split("/")[1], "parent_commit": public["base_commit"],
        "image_url": public["image"], "workdir": "/testbed",
        "problem_statement": public["problem_statement"],
    }])
    runner = TaskRunner(
        task=task, agent_factory=lambda **_: NativeDshAgent(native_task, args),
        llm_config=LLMConfig(model="codex/gpt-6.1-terra", api_key="local-smoke",
                             base_url="http://127.0.0.1:3425/v1"),
        runtime_config=RuntimeConfig(), evaluator=None,
        output_path=args.run_dir.parent / "factory-output",
        max_concurrent=1, max_retries=1, max_steps=60,
    )
    results = await runner.run_all([args.instance_id])
    print(json.dumps({"factory_run_dir": str(runner.run_dir),
                      "errors": [r.error for r in results if r.error],
                      "training_eligible": False}))
    if any(r.error for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
