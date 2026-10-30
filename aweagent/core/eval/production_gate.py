"""Require current, evidence-backed P0 acceptance before batch generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GATES = ('tool_supervision', 'environment', 'request_token_alignment')


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def require_acceptance(receipt_path, *, solver_id: str, project_root: Path, schema_version: int = 1) -> dict:
    """``schema_version`` 2 is the takeover-high chain; a v1 (official low) receipt never covers it."""
    if not receipt_path:
        raise ValueError('P0 acceptance is required before production; run acceptance smoke first')
    receipt = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    if receipt.get('schema_version') != schema_version or receipt.get('solver_id') != solver_id:
        raise ValueError('P0 receipt does not cover this solver entrypoint')
    root = project_root.resolve()
    def verify_file(row):
        path = (root / row['path']).resolve()
        if root not in path.parents or not path.is_file() or digest(path) != row['sha256']:
            raise ValueError('P0 evidence or source changed: ' + row['path'])
    for gate in GATES:
        row = receipt.get('gates', {}).get(gate, {})
        if row.get('passed') is not True or not row.get('evidence'):
            raise ValueError('P0 gate is incomplete: ' + gate)
        for evidence in row['evidence']:
            verify_file(evidence)
    if not receipt.get('sources'):
        raise ValueError('P0 receipt has no frozen implementation sources')
    for source in receipt['sources']:
        verify_file(source)
    return receipt
