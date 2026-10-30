"""Independent ScaleSWE evaluation entry. Never pass its inputs to the solver."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aweagent.core.runtime.config import RuntimeConfig, DockerConfig
from aweagent.core.runtime.docker import DockerRuntime
from aweagent.tasks.scale_swe.task import ScaleSWETask


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tasks", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.source_tasks.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if raw.get("instance_id") == args.instance_id:
                break
        else:
            raise RuntimeError("registered task not found")
    task = ScaleSWETask(instances=[raw])
    instance = task.get_instances()[0]
    # Source schema may store a literal backslash-n; use the same setup as solve.
    instance.setup_commands = [command.removesuffix("\\n") for command in instance.setup_commands]
    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    assert prediction["instance_id"] == instance.id
    runtime = DockerRuntime(RuntimeConfig(
        workdir=instance.workdir, docker=DockerConfig(network="none", pull_policy="never")))
    result = await task.default_evaluator(timeout=1800).evaluate(
        instance, prediction.get("model_patch", ""), runtime)
    output = {
        "instance_id": instance.id, "training_eligible": False,
        "prediction_sha256": hashlib.sha256(args.prediction.read_bytes()).hexdigest(),
        "judge": "ScaleSWEEvaluator / PatchTestEvaluator", "fresh_container": True,
        "network": "none", "result": asdict(result),
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"instance_id": instance.id, "accepted": result.accepted,
                      "error": result.details.get("error"), "receipt": str(args.output)}))


if __name__ == "__main__":
    asyncio.run(main())
