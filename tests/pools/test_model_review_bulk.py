"""Bulk reviewer plumbing: answer parsing, index merging, freshness, usage, and validation.

The bulk stage talks to a network reviewer, so its own logic must be testable without
one: JSON extraction from model prose, snapshot index building, freshness checking
(packet and config changes require rerun, missing-sha old labels never skipped),
wrong packet_id retry without overwrite, per-call usage logging on failure (mocking urllib),
and fresh-only ingestion.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "results"))

import model_review  # noqa: E402
import model_review_bulk  # noqa: E402


class ParseAnswerTest(unittest.TestCase):
    def test_plain_json(self) -> None:
        answer = model_review_bulk._parse_answer('{"packet_id": "MR-B:x", "verdict": "KEEP"}')
        self.assertEqual(answer["verdict"], "KEEP")

    def test_fenced_json_is_unwrapped(self) -> None:
        text = '```json\n{"packet_id": "MR-C1:y", "verdict": "MASK"}\n```'
        self.assertEqual(model_review_bulk._parse_answer(text)["packet_id"], "MR-C1:y")

    def test_garbage_raises(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            model_review_bulk._parse_answer("no json here")


class IndexMergeTest(unittest.TestCase):
    def test_both_indexes_are_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            review = Path(tmp) / "review"
            (review / "bulk").mkdir(parents=True)
            sampled = review / "packet_index.json"
            sampled.write_text(json.dumps({"packets": [
                {"packet_id": "MR-C3:a", "path": str(Path(tmp) / "a.json")}]}),
                encoding="utf-8")
            bulk = review / "bulk" / "bulk_index.json"
            bulk.write_text(json.dumps({"packets": [
                {"packet_id": "MR-B:b", "path": str(Path(tmp) / "b.json")}]}),
                encoding="utf-8")
            with patch.object(model_review, "REVIEW", review):
                paths = model_review.packet_index_paths()
        self.assertEqual(set(paths), {"MR-C3:a", "MR-B:b"})

    def test_missing_indexes_are_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(model_review, "REVIEW", Path(tmp) / "absent"):
                self.assertEqual(model_review.packet_index_paths(), {})


class TargetSelectionTest(unittest.TestCase):
    def test_targets_are_admit_rows_only(self) -> None:
        ledger = {
            "C1:a": {"sample_id": "C1:a", "pool": "C1", "decision": "ADMIT",
                     "instance_id": "a"},
            "C1:b": {"sample_id": "C1:b", "pool": "C1", "decision": "REJECT",
                     "instance_id": "b"},
            "B:c": {"sample_id": "B:c", "pool": "B", "decision": "REVIEW",
                    "instance_id": "c"},
        }
        with patch.object(model_review, "load_ledger", return_value=ledger):
            rows = model_review_bulk.targets()
        self.assertEqual([row["sample_id"] for row in rows], ["C1:a"])

    def test_build_fails_when_target_missing_record(self) -> None:
        ledger = {"C1:a": {"sample_id": "C1:a", "pool": "C1", "decision": "ADMIT",
                           "instance_id": "a"}}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(model_review, "load_ledger", return_value=ledger), \
                patch.object(model_review, "load_records", return_value={}):
            with self.assertRaises(SystemExit) as ctx:
                model_review_bulk.build(bulk_dir=Path(tmp), packets_dir=Path(tmp) / "packets")
            self.assertIn("missing records", str(ctx.exception))

    def test_build_succeeds_with_snapshot_when_records_present(self) -> None:
        ledger = {"C1:a": {"sample_id": "C1:a", "pool": "C1", "decision": "ADMIT",
                           "instance_id": "a"}}
        record = {"instance_id": "a", "messages": [], "patch": ""}
        mock_packet = {
            "packet_id": "MR-C1:a",
            "sample_id": "C1:a",
            "pool": "C1",
            "packet_sha256": "sha_sample_123",
        }
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(model_review, "load_ledger", return_value=ledger), \
                patch.object(model_review, "load_records", return_value={"a": record}), \
                patch.object(model_review, "packet_for", return_value=mock_packet), \
                patch.object(model_review, "packet_path", return_value=Path("MR-C1_a.json")):
            summary = model_review_bulk.build(
                bulk_dir=Path(tmp), packets_dir=Path(tmp) / "packets"
            )
            index = json.loads((Path(tmp) / "bulk_index.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["packets"], 1)
        self.assertIn("snapshot", summary)
        self.assertEqual(index["snapshot"], summary["snapshot"])
        self.assertEqual(index["target_count"], 1)
        self.assertEqual(index["packets"][0]["packet_sha256"], "sha_sample_123")


class FreshnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.packet = {
            "packet_id": "MR-C1:test",
            "sample_id": "C1:test",
            "pool": "C1",
            "packet_sha256": "packet_sha_aaa",
            "candidate_msgs": [1],
            "visible_msgs": [0, 1],
        }
        self.config = model_review_bulk.make_review_config(
            model="deepseek-flash", effort="max", max_tokens=8192
        )
        self.config_sha = model_review_bulk.compute_review_config_sha256(self.config)
        self.valid_sub = {
            "packet_id": "MR-C1:test",
            "packet_sha256": "packet_sha_aaa",
            "review_config_sha256": self.config_sha,
            "verdict": "KEEP",
            "labels": [{"msg": 1, "label": "KEEP",
                        "reason_codes": ["KEEP_EFFECTIVE_EDIT"],
                        "evidence_msgs": [1]}],
            "notes": "all good",
        }

    def test_valid_fresh_submission_is_accepted(self) -> None:
        self.assertTrue(
            model_review_bulk.is_submission_fresh(self.valid_sub, self.packet, self.config_sha)
        )

    def test_packet_change_requires_rerun(self) -> None:
        # If the packet content changed, packet_sha256 differs
        altered_packet = dict(self.packet, packet_sha256="packet_sha_bbb")
        self.assertFalse(
            model_review_bulk.is_submission_fresh(self.valid_sub, altered_packet, self.config_sha)
        )

    def test_config_change_requires_rerun(self) -> None:
        # If model/effort/max_tokens/prompt changed, config_sha differs
        other_config = model_review_bulk.make_review_config(
            model="deepseek-flash", effort="high", max_tokens=8192
        )
        other_config_sha = model_review_bulk.compute_review_config_sha256(other_config)
        self.assertFalse(
            model_review_bulk.is_submission_fresh(self.valid_sub, self.packet, other_config_sha)
        )

    def test_old_submission_without_sha_never_skipped(self) -> None:
        # Old labels from sample stage or legacy runs without sha must not skip
        legacy_sub_no_packet_sha = dict(self.valid_sub)
        legacy_sub_no_packet_sha.pop("packet_sha256")
        self.assertFalse(
            model_review_bulk.is_submission_fresh(legacy_sub_no_packet_sha, self.packet, self.config_sha)
        )

        legacy_sub_no_config_sha = dict(self.valid_sub)
        legacy_sub_no_config_sha.pop("review_config_sha256")
        self.assertFalse(
            model_review_bulk.is_submission_fresh(legacy_sub_no_config_sha, self.packet, self.config_sha)
        )

    def test_invalid_schema_never_skipped(self) -> None:
        # Malformed submission (e.g. candidate message missing) must not be considered fresh
        malformed = dict(self.valid_sub, labels=[])
        self.assertFalse(
            model_review_bulk.is_submission_fresh(malformed, self.packet, self.config_sha)
        )

    def test_run_selection_skips_only_fresh_and_keeps_stale_in_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bulk_dir = Path(tmp) / "bulk"
            packets_dir = bulk_dir / "packets"
            packets_dir.mkdir(parents=True)

            p1 = dict(self.packet, packet_id="MR-C1:fresh", sample_id="C1:fresh", packet_sha256="sha_p1")
            p2 = dict(self.packet, packet_id="MR-C1:stale", sample_id="C1:stale", packet_sha256="sha_p2")
            p3 = dict(self.packet, packet_id="MR-C1:nosha", sample_id="C1:nosha", packet_sha256="sha_p3")

            p1_file = packets_dir / "p1.json"
            p2_file = packets_dir / "p2.json"
            p3_file = packets_dir / "p3.json"
            p1_file.write_text(json.dumps(p1), encoding="utf-8")
            p2_file.write_text(json.dumps(p2), encoding="utf-8")
            p3_file.write_text(json.dumps(p3), encoding="utf-8")

            index = {
                "review_version": "v1",
                "snapshot": "snap_test",
                "packets": [
                    {"packet_id": "MR-C1:fresh", "path": str(p1_file), "packet_sha256": "sha_p1"},
                    {"packet_id": "MR-C1:stale", "path": str(p2_file), "packet_sha256": "sha_p2"},
                    {"packet_id": "MR-C1:nosha", "path": str(p3_file), "packet_sha256": "sha_p3"},
                ],
            }
            (bulk_dir / "bulk_index.json").write_text(json.dumps(index), encoding="utf-8")

            # p1 has valid fresh submission
            sub1 = dict(self.valid_sub, packet_id="MR-C1:fresh", packet_sha256="sha_p1", review_config_sha256=self.config_sha)
            # p2 has submission with mismatched config_sha (stale)
            sub2 = dict(self.valid_sub, packet_id="MR-C1:stale", packet_sha256="sha_p2", review_config_sha256="wrong_config_sha")
            # p3 has legacy submission with no sha
            sub3 = {"packet_id": "MR-C1:nosha", "verdict": "KEEP", "labels": []}

            submissions_file = bulk_dir / "submissions.jsonl"
            submissions_file.write_text("\n".join(json.dumps(s) for s in [sub1, sub2, sub3]) + "\n", encoding="utf-8")

            mock_answers = {
                "MR-C1:stale": {"packet_id": "MR-C1:stale", "verdict": "KEEP", "labels": []},
                "MR-C1:nosha": {"packet_id": "MR-C1:nosha", "verdict": "KEEP", "labels": []},
            }
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake_key"}), \
                    patch.object(model_review_bulk, "review_one", side_effect=lambda p, *a, **k: mock_answers[p["packet_id"]]):
                summary = model_review_bulk.run(
                    model="deepseek-flash",
                    effort="max",
                    max_tokens=8192,
                    bulk_dir=bulk_dir,
                    submissions_path=submissions_file,
                )

            # Total pending should be 2: stale and nosha remain pending, while fresh was skipped
            self.assertEqual(summary["pending_total"], 2)
            self.assertEqual(summary["accepted"], 2)



class WrongPacketIdTest(unittest.TestCase):
    def test_wrong_packet_id_not_overwritten_and_retries(self) -> None:
        packet = {
            "packet_id": "MR-TARGET:1",
            "sample_id": "TARGET:1",
            "pool": "C1",
            "packet_sha256": "sha_111",
            "candidate_msgs": [1],
            "visible_msgs": [0, 1],
        }

        call_count = 0
        recorded_messages = []

        def mock_post(payload: dict, api_key: str, base_url: str = "", timeout: float = 0.0) -> dict:
            nonlocal call_count, recorded_messages
            call_count += 1
            recorded_messages.append(list(payload["messages"]))
            if call_count == 1:
                # First attempt returns a wrong packet_id
                wrong_content = json.dumps({
                    "packet_id": "MR-WRONG:999",
                    "verdict": "KEEP",
                    "labels": [{"msg": 1, "label": "KEEP",
                                "reason_codes": ["KEEP_EFFECTIVE_EDIT"],
                                "evidence_msgs": [1]}],
                    "notes": "wrong id on purpose",
                })
                return {"choices": [{"message": {"content": wrong_content}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
            else:
                # Second attempt corrects the packet_id
                correct_content = json.dumps({
                    "packet_id": "MR-TARGET:1",
                    "verdict": "KEEP",
                    "labels": [{"msg": 1, "label": "KEEP",
                                "reason_codes": ["KEEP_EFFECTIVE_EDIT"],
                                "evidence_msgs": [1]}],
                    "notes": "corrected id",
                })
                return {"choices": [{"message": {"content": correct_content}}],
                        "usage": {"prompt_tokens": 120, "completion_tokens": 60}}

        with tempfile.TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "usage.jsonl"
            with patch.object(model_review_bulk, "_post", side_effect=mock_post):
                answer = model_review_bulk.review_one(
                    packet=packet,
                    api_key="fake-key",
                    model="deepseek-flash",
                    effort="max",
                    usage_path=usage_path,
                    sleep_fn=lambda _: None,
                )

        self.assertEqual(call_count, 2)
        # Verify the retry prompt told the model about packet_id mismatch
        second_attempt_messages = recorded_messages[1]
        user_retry_prompt = second_attempt_messages[-1]["content"]
        self.assertIn("packet_id mismatch", user_retry_prompt)
        # Verify final answer preserves correct id and receipts
        self.assertEqual(answer["packet_id"], "MR-TARGET:1")
        self.assertEqual(answer["effort"], "max")
        self.assertEqual(answer["packet_sha256"], "sha_111")
        self.assertIn("review_config", answer)

    def test_persistent_wrong_packet_id_raises_runtime_error(self) -> None:
        packet = {
            "packet_id": "MR-TARGET:1",
            "sample_id": "TARGET:1",
            "pool": "C1",
            "packet_sha256": "sha_111",
            "candidate_msgs": [1],
            "visible_msgs": [0, 1],
        }

        wrong_content = json.dumps({
            "packet_id": "MR-ALWAYS-WRONG",
            "verdict": "KEEP",
            "labels": [{"msg": 1, "label": "KEEP",
                        "reason_codes": ["KEEP_EFFECTIVE_EDIT"],
                        "evidence_msgs": [1]}],
        })

        def mock_post(*_args, **_kwargs) -> dict:
            return {"choices": [{"message": {"content": wrong_content}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20}}

        with tempfile.TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "usage.jsonl"
            with patch.object(model_review_bulk, "_post", side_effect=mock_post):
                with self.assertRaises(RuntimeError) as ctx:
                    model_review_bulk.review_one(
                        packet=packet,
                        api_key="fake-key",
                        usage_path=usage_path,
                        sleep_fn=lambda _: None,
                    )
            self.assertIn("packet_id mismatch", str(ctx.exception))


class UsageLoggingTest(unittest.TestCase):
    def test_failed_usage_logged_per_call_and_no_keys_recorded(self) -> None:
        packet = {
            "packet_id": "MR-FAIL:1",
            "sample_id": "FAIL:1",
            "pool": "C1",
            "packet_sha256": "sha_fail",
            "candidate_msgs": [1],
            "visible_msgs": [0, 1],
        }

        # Mock urllib.request.urlopen returning invalid JSON but with valid usage
        call_count = 0
        fake_api_key = "sk-secret-never-to-be-logged-123456"

        def mock_urlopen(request: urllib.request.Request, timeout: float = 0.0):
            nonlocal call_count
            call_count += 1
            # Verify request header had bearer token
            self.assertIn(f"Bearer {fake_api_key}", request.headers.get("Authorization", ""))
            response_data = {
                "choices": [{"message": {"content": "not a json answer"}}],
                "usage": {
                    "prompt_tokens": 150 + call_count,
                    "completion_tokens": 30 + call_count,
                    "total_tokens": 180 + 2 * call_count,
                },
            }
            body = json.dumps(response_data).encode("utf-8")
            resp = MagicMock()
            resp.read.return_value = body
            resp.__enter__.return_value = resp
            resp.__exit__.return_value = False
            return resp

        with tempfile.TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "usage.jsonl"
            with patch("urllib.request.urlopen", side_effect=mock_urlopen):
                with self.assertRaises(RuntimeError):
                    model_review_bulk.review_one(
                        packet=packet,
                        api_key=fake_api_key,
                        usage_path=usage_path,
                        sleep_fn=lambda _: None,
                    )

            # Check usage file: should have 3 lines (one for each attempt)
            self.assertTrue(usage_path.exists())
            lines = usage_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)

            for idx, line in enumerate(lines, 1):
                rec = json.loads(line)
                self.assertEqual(rec["packet_id"], "MR-FAIL:1")
                self.assertEqual(rec["prompt_tokens"], 150 + idx)
                self.assertEqual(rec["completion_tokens"], 30 + idx)
                # Confirm secret key is nowhere in the log
                self.assertNotIn(fake_api_key, line)
                self.assertNotIn("sk-secret", line)


class IngestFreshSelectionTest(unittest.TestCase):
    def test_ingest_selects_last_fresh_submission_only(self) -> None:
        packet = {
            "packet_id": "MR-C1:a",
            "sample_id": "C1:a",
            "pool": "C1",
            "packet_sha256": "packet_sha_123",
            "candidate_msgs": [1],
            "visible_msgs": [0, 1],
        }
        config = model_review_bulk.make_review_config(
            model="deepseek-flash", effort="max", max_tokens=8192
        )
        config_sha = model_review_bulk.compute_review_config_sha256(config)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bulk_dir = tmp_path / "bulk"
            packets_dir = bulk_dir / "packets"
            packets_dir.mkdir(parents=True)
            p_file = packets_dir / "MR-C1_a.json"
            p_file.write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")

            index = {
                "review_version": "v1",
                "snapshot": "snap123",
                "packets": [{"packet_id": "MR-C1:a", "path": str(p_file)}],
            }
            index_file = bulk_dir / "bulk_index.json"
            index_file.write_text(json.dumps(index), encoding="utf-8")

            submissions_file = bulk_dir / "submissions.jsonl"
            # 1: Stale submission with old packet sha
            sub_stale = {
                "packet_id": "MR-C1:a",
                "packet_sha256": "old_sha",
                "review_config_sha256": config_sha,
                "verdict": "KEEP",
                "labels": [{"msg": 1, "label": "KEEP", "reason_codes": ["KEEP_EFFECTIVE_EDIT"], "evidence_msgs": [1]}],
                "notes": "stale_note",
            }
            # 2: Fresh submission 1
            sub_fresh1 = {
                "packet_id": "MR-C1:a",
                "packet_sha256": "packet_sha_123",
                "review_config_sha256": config_sha,
                "verdict": "KEEP",
                "labels": [{"msg": 1, "label": "KEEP", "reason_codes": ["KEEP_EFFECTIVE_EDIT"], "evidence_msgs": [1]}],
                "notes": "fresh_1",
            }
            # 3: Fresh submission 2 (latest)
            sub_fresh2 = {
                "packet_id": "MR-C1:a",
                "packet_sha256": "packet_sha_123",
                "review_config_sha256": config_sha,
                "verdict": "MASK",
                "labels": [{"msg": 1, "label": "MASK", "reason_codes": ["MASK_FALSE_CLAIM"], "evidence_msgs": [1]}],
                "notes": "fresh_2_latest",
            }

            submissions_file.write_text(
                "\n".join(json.dumps(s) for s in [sub_stale, sub_fresh1, sub_fresh2]) + "\n",
                encoding="utf-8"
            )

            ingested_records = []

            def mock_mr_ingest(file_path: Path) -> dict:
                nonlocal ingested_records
                ingested_records = [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines()]
                return {"accepted": len(ingested_records), "rejected": 0, "total_labels": len(ingested_records)}

            with patch.object(model_review, "ingest", side_effect=mock_mr_ingest):
                result = model_review_bulk.ingest(
                    model="deepseek-flash",
                    effort="max",
                    max_tokens=8192,
                    submissions_path=submissions_file,
                    index_path=index_file,
                )

            self.assertEqual(result["fresh_selected"], 1)
            self.assertEqual(len(ingested_records), 1)
            # Must be fresh_2_latest, stale and earlier fresh must not override
            self.assertEqual(ingested_records[0]["notes"], "fresh_2_latest")
            self.assertEqual(ingested_records[0]["verdict"], "MASK")


if __name__ == "__main__":
    unittest.main()
