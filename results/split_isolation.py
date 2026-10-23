"""Split isolation and benchmark family protection rules.

Enforces strict train/test isolation per AGENTS.md:
  - Benchmark evaluation families (e.g. SWE-bench Verified) must never leak into
    training pools (Pool A, Pool B, Pool C) or teacher continuation seeds.
  - Fail-closed: missing or invalid protection config immediately halts/rejects.
  - Case-insensitive comparison (casefold + strip) across all family handles.
"""
from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "data" / "protected_swe_families.json"
)

ISOLATION_REASONS = frozenset({"validation_family", "benchmark_family", "unknown_task"})


class SplitIsolationConfigError(Exception):
    """Raised when the split isolation configuration is missing or invalid."""
    pass


class ProtectedFamiliesConfig(NamedTuple):
    dataset: str
    split: str
    source_count: int
    source_agent_view_sha256: str
    families: frozenset[str]
    raw_sha256: str
    path: str


def normalize_family(family: str | None) -> str:
    """Normalize a repository family handle by stripping whitespace and casefolding."""
    if not family:
        return ""
    return family.strip().casefold()


def load_protected_families(config_path: Path | str | None = None) -> ProtectedFamiliesConfig:
    """Load and validate the protected SWE benchmark families configuration.

    Fail-closed: raises SplitIsolationConfigError if the file is missing,
    cannot be parsed, or lacks required fields/non-empty families.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise SplitIsolationConfigError(f"Protected families configuration not found: {path}")

    try:
        raw_bytes = path.read_bytes()
        digest = sha256(raw_bytes).hexdigest()
        data = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise SplitIsolationConfigError(
            f"Failed to read/parse protected families config at {path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise SplitIsolationConfigError(
            f"Config at {path} must be a JSON object, got {type(data).__name__}"
        )

    families_raw = data.get("protected_families")
    if not isinstance(families_raw, list) or not families_raw:
        raise SplitIsolationConfigError(
            f"Config at {path} must contain non-empty 'protected_families' list"
        )

    norm_families = set()
    for fam in families_raw:
        if not isinstance(fam, str) or not re.fullmatch(r"[^/\s]+/[^/\s]+", fam.strip()):
            raise SplitIsolationConfigError(f"Invalid family entry in {path}: {fam!r}")
        norm_families.add(normalize_family(fam))

    # Basic metadata validation
    dataset = data.get("dataset")
    split = data.get("split")
    source_count = data.get("source_count")
    source_agent_view_sha256 = data.get("source_agent_view_sha256")
    if (dataset not in {"SWE-bench/SWE-bench_Verified", "princeton-nlp/SWE-bench_Verified"}
            or split != "test" or type(source_count) is not int or source_count != 500
            or not isinstance(source_agent_view_sha256, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", source_agent_view_sha256)):
        raise SplitIsolationConfigError(f"Config at {path} has invalid Verified test metadata")

    return ProtectedFamiliesConfig(
        dataset=dataset,
        split=split,
        source_count=source_count,
        source_agent_view_sha256=source_agent_view_sha256,
        families=frozenset(norm_families),
        raw_sha256=digest,
        path=str(path),
    )


def is_benchmark_family(
    family: str | None,
    protected_families: set[str] | frozenset[str] | None = None,
    config_path: Path | str | None = None,
) -> bool:
    """Check if the given family belongs to the protected benchmark families.

    Fail-closed: if protected_families is not provided and config cannot be loaded,
    raises SplitIsolationConfigError.
    """
    if protected_families is None:
        cfg = load_protected_families(config_path)
        protected_families = cfg.families

    norm = normalize_family(family)
    if not norm:
        return False
    return norm in protected_families


def evaluate_benchmark_gate(
    family: str | None,
    protected_families: set[str] | frozenset[str] | None = None,
    config_path: Path | str | None = None,
) -> tuple[bool, str | None]:
    """Pure gate function to evaluate benchmark family admission.

    Returns:
        (passed: bool, rejection_reason: str | None)
        If rejected, rejection_reason is 'benchmark_family'.
    Fail-closed: raises SplitIsolationConfigError if config is invalid.
    """
    if is_benchmark_family(family, protected_families=protected_families, config_path=config_path):
        return False, "benchmark_family"
    return True, None


def is_split_quarantined(
    family: str | None,
    rejected_reasons: list[str] | None = None,
    protected_families: set[str] | frozenset[str] | None = None,
    validation_families: set[str] | frozenset[str] | None = None,
    config_path: Path | str | None = None,
) -> bool:
    """Determine whether an instance is quarantined by split isolation.

    Triggers when:
      1. rejected_reasons contains any isolation gate ('benchmark_family', 'validation_family', 'unknown_task')
      2. family is in protected benchmark families (case-insensitive)
      3. family is in validation_families (case-insensitive)
      4. family is empty/invalid
    Fail-closed: raises SplitIsolationConfigError if config is missing/invalid when resolving protected_families.
    """
    if rejected_reasons:
        for r in rejected_reasons:
            if r in ISOLATION_REASONS:
                return True

    norm = normalize_family(family)
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", norm):
        return True

    if validation_families:
        val_norm = {normalize_family(f) for f in validation_families}
        if norm in val_norm:
            return True

    return is_benchmark_family(norm, protected_families=protected_families, config_path=config_path)
