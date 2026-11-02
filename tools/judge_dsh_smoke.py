"""Independent ScaleSWE evaluation entry. Never pass its inputs to the solver."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aweagent.core.eval import workspace_hardening
from aweagent.core.runtime.config import RuntimeConfig, DockerConfig
from aweagent.core.runtime.docker import DockerRuntime
from aweagent.tasks.scale_swe.evaluator import ScaleSWEEvaluator
from aweagent.tasks.scale_swe.task import ScaleSWETask


# Use Git's normal ignored-file policy while checking tracked and untracked changes.
STATUS_COMMAND = ("git status --porcelain=v1 -z --untracked-files=all "
                  "--no-renames --ignore-submodules=none")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status_counts(output):
    counts = {"tracked": 0, "untracked": 0}
    if output and not output.endswith("\0"):
        raise RuntimeError("invalid judge status output")
    for record in output.split("\0"):
        if not record:
            continue
        if len(record) < 4 or record[2] != " ":
            raise RuntimeError("invalid judge status output")
        if record[:2] == "!!":
            raise RuntimeError("invalid judge status output")
        kind = "untracked" if record[:2] == "??" else "tracked"
        counts[kind] += 1
    return counts


def committed_change_counts(output):
    counts = {"added": 0, "modified": 0, "deleted": 0, "other": 0}
    if not output:
        return counts
    records = output.split("\0")
    if records[-1] or len(records) % 2 != 1:
        raise RuntimeError("invalid judge change output")
    for status, filename in zip(records[0:-1:2], records[1:-1:2]):
        if len(status) != 1 or not filename:
            raise RuntimeError("invalid judge change output")
        counts[{"A": "added", "M": "modified", "D": "deleted"}.get(status, "other")] += 1
    return counts


class JudgeRuntime(DockerRuntime):
    """Record the actual private judge container, rather than asserting freshness."""

    def __init__(self, config, solver_environment):
        super().__init__(config)
        self.solver_environment = solver_environment
        self.evidence = {"independent": False}

    async def create_session(self, image=None, **kwargs):
        session = await super().create_session(image, **kwargs)
        info = session._container.attrs
        self.evidence.update(container_id=session._container.id, image_id=info["Image"],
                             network=info["HostConfig"]["NetworkMode"],
                             bind_mounts=[m["Destination"] for m in info.get("Mounts", [])
                                          if m["Type"] == "bind"])
        independent = (self.evidence["network"] == "none" and not self.evidence["bind_mounts"]
                       and session._container.id != self.solver_environment.get("container_id")
                       and (not self.solver_environment.get("image_id")
                            or info["Image"] == self.solver_environment["image_id"]))
        if not independent:
            await session.close()
            raise RuntimeError("judge container does not satisfy isolated image/network identity")
        self.evidence["independent"] = True
        return session

    def cleanup_verified(self):
        container_id = self.evidence.get("container_id")
        if not container_id:
            return False
        result = subprocess.run(["docker", "container", "inspect", container_id],
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        return result.returncode != 0 and "no such" in (result.stdout + result.stderr).lower()


class JudgeEvaluator(ScaleSWEEvaluator):
    def __init__(self, timeout=1800):
        super().__init__(timeout=timeout)
        self.evidence = self.empty_evidence()

    @staticmethod
    def empty_evidence():
        return {"fresh_before": False, "diagnostics": {
            "stage": "not_started", "base_matched": None, "base_ancestor": None,
            "tree_matches_base": None, "initial_status": None, "prepared_status": None,
            "committed_changes": None, "source_copies": {
                "passed": None, "git_passed": None, "command_succeeded": None,
                "modules": None, "external": None, "unreadable": None,
            },
        }}

    async def evaluate(self, instance, patch, runtime):
        self.evidence = self.empty_evidence()
        return await super().evaluate(instance, patch, runtime)

    async def workspace_status(self, instance, session):
        command = (STATUS_COMMAND + " && git submodule --quiet foreach --recursive "
                   + shlex.quote(STATUS_COMMAND))
        result = await session.execute(command, cwd=instance.workdir)
        if not result.success:
            raise RuntimeError("judge workspace status could not be verified")
        return status_counts(result.stdout)

    async def audit_source_copies(self, instance, session, diagnostics):
        source = Path(workspace_hardening.__file__).read_text(encoding="utf-8")
        command = ("python -c " + shlex.quote(source) + " --root " + shlex.quote(instance.workdir)
                   + " --base " + shlex.quote(instance.base_commit) + " --audit-copies")
        result = await session.execute(command, cwd=instance.workdir, timeout=300)
        diagnostics["source_copies"]["command_succeeded"] = result.success
        try:
            receipt = json.loads(result.stdout)
            copies = receipt["source_copies"]
            modules, external, unreadable = (copies[key] for key in ("modules", "external_copies", "unreadable_paths"))
            if not all(isinstance(value, list) for value in (modules, external, unreadable)):
                raise ValueError("invalid audit counts")
            passed = (receipt.get("source_copies_passed") is True
                      and bool(modules) and not external and not unreadable)
            diagnostics["source_copies"].update(passed=passed, git_passed=receipt.get("git_passed") is True,
                                                  modules=len(modules), external=len(external),
                                                  unreadable=len(unreadable))
        except (ValueError, TypeError, KeyError):
            raise RuntimeError("judge source-copy audit could not be verified") from None
        if not result.success or receipt.get("git_passed") is not True or not passed:
            raise RuntimeError("judge source-copy audit did not pass")

    async def pre_patch_setup(self, instance, session):
        self.evidence = self.empty_evidence()
        diagnostics = self.evidence["diagnostics"]
        diagnostics["stage"] = "initial_identity"
        before = await session.execute("git rev-parse HEAD HEAD^{tree}", cwd=instance.workdir)
        identity = before.stdout.strip().splitlines()
        if before.success and len(identity) == 2:
            diagnostics["base_matched"] = identity[0] == instance.base_commit
        if diagnostics["base_matched"] is not True:
            raise RuntimeError("judge did not checkout the declared before commit")
        self.evidence.update(base_commit=identity[0], base_tree=identity[1])

        diagnostics["stage"] = "initial_status"
        diagnostics["initial_status"] = await self.workspace_status(instance, session)
        if any(diagnostics["initial_status"].values()):
            raise RuntimeError("judge before workspace is not clean")

        diagnostics["stage"] = "setup"
        await super().pre_patch_setup(instance, session)
        base = shlex.quote(instance.base_commit)
        diagnostics["stage"] = "prepared_state"
        prepared = await session.execute("git rev-parse HEAD HEAD^{tree}", cwd=instance.workdir)
        after = prepared.stdout.strip().splitlines()
        if not prepared.success or len(after) != 2:
            raise RuntimeError("judge prepared identity could not be verified")
        self.evidence.update(prepared_commit=after[0], prepared_tree=after[1])
        diagnostics["tree_matches_base"] = after[1] == identity[1]
        diagnostics["prepared_status"] = await self.workspace_status(instance, session)
        changes = await session.execute(
            f"git diff --name-status --no-renames -z {base} HEAD --", cwd=instance.workdir)
        if not changes.success:
            raise RuntimeError("judge prepared changes could not be verified")
        diagnostics["committed_changes"] = committed_change_counts(changes.stdout)
        ancestor = await session.execute(f"git merge-base --is-ancestor {base} HEAD", cwd=instance.workdir)
        if ancestor.exit_code not in (0, 1):
            raise RuntimeError("judge base ancestry could not be verified")
        diagnostics["base_ancestor"] = ancestor.exit_code == 0
        if not diagnostics["base_ancestor"]:
            diagnostics["stage"] = "prepared_tree"
            raise RuntimeError("judge setup changed the declared before source tree")

        diagnostics["stage"] = "source_copy_audit"
        await self.audit_source_copies(instance, session, diagnostics)
        diagnostics["stage"] = "prepared_tree"
        if (not diagnostics["tree_matches_base"] or any(diagnostics["prepared_status"].values())
                or any(diagnostics["committed_changes"].values())):
            raise RuntimeError("judge setup changed the declared before source tree")
        diagnostics["stage"] = "complete"
        self.evidence["fresh_before"] = True


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
    prediction_bytes = args.prediction.read_bytes()
    prediction = json.loads(prediction_bytes)
    if prediction.get("instance_id") != instance.id:
        raise ValueError("prediction instance does not match the judge task")
    run_root = args.prediction.parent
    protocol_path, provenance_path = run_root / "protocol.json", run_root / "request-provenance.jsonl"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {}
    environment_path = run_root / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.exists() else {}
    bound_files = {p: sha(p) for p in (protocol_path, provenance_path, environment_path) if p.is_file()}
    task_source_sha256 = sha(args.source_tasks)
    if protocol.get("schema_version") == 2 and (not provenance_path.is_file() or environment.get("passed") is not True):
        raise ValueError("v2 judge requires completed solver provenance and environment evidence")
    runtime = JudgeRuntime(RuntimeConfig(
        workdir=instance.workdir, docker=DockerConfig(network="none", pull_policy="never")), environment)
    evaluator = JudgeEvaluator()
    result = await evaluator.evaluate(instance, prediction.get("model_patch", ""), runtime)
    if args.prediction.read_bytes() != prediction_bytes:
        raise ValueError("prediction changed while judging")
    if any(sha(path) != expected for path, expected in bound_files.items()):
        raise ValueError("solver evidence changed while judging")
    cleanup_verified = runtime.cleanup_verified()
    output = {
        "schema_version": 2,
        "instance_id": instance.id, "training_eligible": False,
        "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "judge": "ScaleSWEEvaluator / PatchTestEvaluator",
        "independent": runtime.evidence["independent"],
        "fresh_before": evaluator.evidence["fresh_before"],
        "fresh_container": runtime.evidence["independent"],
        "cleanup_verified": cleanup_verified,
        "environment": runtime.evidence, "before": evaluator.evidence,
        "synthetic": protocol.get("synthetic", False),
        "judge_source_sha256": sha(Path(__file__)),
        "task_source_sha256": task_source_sha256,
        "network": "none", "result": asdict(result),
    }
    if protocol_path.is_file():
        output["protocol_sha256"] = sha(protocol_path)
    if provenance_path.is_file():
        output["provenance_sha256"] = sha(provenance_path)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"instance_id": instance.id, "accepted": result.accepted,
                      "independent": output["independent"], "fresh_before": output["fresh_before"],
                      "cleanup_verified": cleanup_verified, "receipt": str(args.output),
                      "diagnostics": evaluator.evidence["diagnostics"]}))
    if not cleanup_verified:
        raise SystemExit("judge cleanup could not be verified; receipt retained")


if __name__ == "__main__":
    asyncio.run(main())
