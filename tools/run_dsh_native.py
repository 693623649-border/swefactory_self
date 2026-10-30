"""Factory TaskRunner entry using the local_train eval DSH solver.

Input is an already exported public task fixture, never the hidden task record.
This opt-in smoke entry leaves the historical Python dsh_compat path unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
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
from aweagent.core.eval.production_gate import require_acceptance


def acceptance_identity(args, driver):
    assets = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {'dsh': driver.runtime_identity(args.dsh_repo), 'model': args.model,
            'upstream': args.upstream.rstrip('/'), 'profile': args.profile,
            'profile_sha256': sha(args.eval_root/'profiles'/args.profile/'cordis.patch.yml'),
            'budget_sha256': sha(args.budget),
            'student_tokenizer_assets': {name: sha(args.student_model_dir/name) for name in assets}}


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
            "production", args.dsh_repo, args.upstream, profile_name=args.profile,
            model=args.model, budget_path=args.budget, api_key_file=args.api_key_file,
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
            metadata={"training_eligible": False, "native_receipts": str(args.run_dir),
                      "generation_kind": args.mode,
                      "production_acceptance_sha256": (hashlib.sha256(args.acceptance.read_bytes()).hexdigest()
                                                       if args.mode == 'production' else None)},
        )


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--dsh-repo", type=Path, required=True)
    parser.add_argument("--upstream", default="https://api.deepseek.com/v1")
    parser.add_argument("--profile", default="swe-agent-v3-deepseek-p0")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--budget", type=Path)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--mode", choices=("smoke", "production"), default="smoke")
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--student-model-dir", type=Path, default=Path('E:/Deeplearning/local_train/model/Qwen3.5-4B'))
    args = parser.parse_args()
    for key in ("task_file", "run_dir", "eval_root", "dsh_repo"):
        setattr(args, key, getattr(args, key).resolve())
    sys.path.insert(0, str(args.eval_root / "swe"))
    import native_task

    args.budget = args.budget.resolve() if args.budget else args.eval_root/'config/budget-deepseek-p0.json'
    public = native_task.load_task_row(args.instance_id, args.task_file)
    if args.mode == 'production':
        try:
            receipt = require_acceptance(args.acceptance, solver_id='dsh-native-v1', project_root=args.eval_root.parent)
        except (ValueError, OSError, KeyError) as error:
            raise SystemExit(str(error)) from error
        if receipt.get('runtime_identity') != acceptance_identity(args, native_task):
            raise SystemExit('P0 runtime/profile/tokenizer identity changed; revalidate before production')
    elif args.validate_only:
        parser.error('--validate-only requires --mode production')
    if args.validate_only:
        print(json.dumps({'ready': True, 'solver_id': 'dsh-native-v1', 'model': args.model,
                          'new_model_calls': 0}))
        return
    if args.upstream.startswith('https://api.deepseek.com') and args.api_key_file is None:
        parser.error('DeepSeek requires --api-key-file; the secret is never embedded in a profile')

    task = NativeTask(instances=[{
        "instance_id": public["instance_id"], "user": public["repo"].split("/")[0],
        "repo": public["repo"].split("/")[1], "parent_commit": public["base_commit"],
        "image_url": public["image"], "workdir": "/testbed",
        "problem_statement": public["problem_statement"],
    }])
    runner = TaskRunner(
        task=task, agent_factory=lambda **_: NativeDshAgent(native_task, args),
        llm_config=LLMConfig(model=args.model, api_key="local-smoke", base_url=args.upstream),
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
