"""Require current, evidence-backed P0 acceptance before batch generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GATES = ('tool_supervision', 'environment', 'request_token_alignment')
# Schema-2 receipts hash frozen sources by this rule so a checkout's line-ending
# conversion (core.autocrlf, eol attributes) cannot change their identity:
# Git's text test (no NUL in the first 8000 bytes), then CRLF -> LF, then sha256.
# Binary sources and all gate evidence keep raw-byte sha256.
SOURCE_DIGEST_RULE = 'sha256-git-text-lf-v1'


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def source_digest(path: Path) -> str:
    data = path.read_bytes()
    if b'\0' not in data[:8000]:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def require_acceptance(receipt_path, *, solver_id: str, project_root: Path, schema_version: int = 1) -> dict:
    """``schema_version`` 2 is the takeover-high chain; a v1 (official low) receipt never covers it."""
    if not receipt_path:
        raise ValueError('P0 acceptance is required before production; run acceptance smoke first')
    receipt = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    if receipt.get('schema_version') != schema_version or receipt.get('solver_id') != solver_id:
        raise ValueError('P0 receipt does not cover this solver entrypoint')
    root = project_root.resolve()
    if schema_version >= 2 and receipt.get('source_digest_rule') != SOURCE_DIGEST_RULE:
        raise ValueError('P0 receipt does not declare source digest rule ' + SOURCE_DIGEST_RULE)
    source_hash = source_digest if schema_version >= 2 else digest
    def verify_file(row, hash_file=digest):
        path = (root / row['path']).resolve()
        if root not in path.parents or not path.is_file() or hash_file(path) != row['sha256']:
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
        verify_file(source, source_hash)
    return receipt
