#!/usr/bin/env python3
"""Tests for _snapshot_lib.py — snapshot generation/compression (ADR-078 Increment 3).

Includes the REGRESSION guard for the inc3 adversarial-verification finding:
generate_snapshot_md / _extract_quality_gate_state call _validation_lib helpers
(validate_step_output, parse_review_verdict) that the extraction initially failed
to import, producing swallowed NameErrors that silently dropped IMMORTAL snapshot
sections. These tests exercise those exact branches so the gap stays closed.

Run: python3 -m pytest _test_snapshot_lib.py -v
  or: python3 _test_snapshot_lib.py
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import _snapshot_lib as snap


class TestValidationDepsImported(unittest.TestCase):
    """Regression (ADR-078 inc3): the cross-module validators snapshot calls must
    be importable in _snapshot_lib's namespace, or call-time NameErrors silently
    drop the Anti-Skip Guard and Quality Gate snapshot sections."""

    def test_validation_helpers_bound(self):
        self.assertTrue(hasattr(snap, "validate_step_output"),
                        "validate_step_output must be imported from _validation_lib")
        self.assertTrue(hasattr(snap, "parse_review_verdict"),
                        "parse_review_verdict must be imported from _validation_lib")


class TestExtractQualityGateState(unittest.TestCase):
    def setUp(self):
        self.proj = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.proj)

    def test_no_gate_logs_returns_empty(self):
        self.assertEqual(snap._extract_quality_gate_state(str(self.proj)), [])

    def test_review_log_resolves_parse_review_verdict(self):
        # REGRESSION: this path calls parse_review_verdict OUTSIDE any try/except,
        # so a missing import raises NameError directly from this function.
        (self.proj / "review-logs").mkdir()
        (self.proj / "review-logs" / "step-5-review.md").write_text(
            "# Review — Step 5\n\n## Verdict\n\nPASS\n\nCritical: 0\nWarning: 1\n",
            encoding="utf-8")
        lines = snap._extract_quality_gate_state(str(self.proj))
        self.assertTrue(any("Step 5" in ln for ln in lines),
                        f"expected latest-step line, got {lines}")

    def test_pacs_log_extracts_score(self):
        (self.proj / "pacs-logs").mkdir()
        (self.proj / "pacs-logs" / "step-3-pacs.md").write_text(
            "# pACS — Step 3\n\npACS = min(F, C, L) = 78\n", encoding="utf-8")
        lines = snap._extract_quality_gate_state(str(self.proj))
        self.assertTrue(any("Step 3" in ln for ln in lines))
        self.assertTrue(any("78" in ln for ln in lines))


class TestGenerateSnapshotMd(unittest.TestCase):
    def setUp(self):
        self.proj = Path(tempfile.mkdtemp())
        (self.proj / ".claude").mkdir()

    def tearDown(self):
        shutil.rmtree(self.proj)

    def _entries(self):
        return [
            {"type": "user_message", "content": "build the feature", "timestamp": "t1"},
            {"type": "assistant_text", "content": "on it", "timestamp": "t2"},
        ]

    def test_basic_snapshot_is_string_with_header(self):
        md = snap.generate_snapshot_md("s1", "Stop", str(self.proj), self._entries())
        self.assertIsInstance(md, str)
        self.assertIn(snap.SNAPSHOT_SECTION_MARKERS["header"], md)

    def test_antiskip_section_runs_validate_step_output(self):
        # REGRESSION: autopilot outputs trigger the validate_step_output loop.
        # Without the _validation_lib import the loop dies (swallowed NameError)
        # and the header appears with ZERO [OK]/[FAIL] marks.
        out = self.proj / "out-step-1.md"
        out.write_text("# Step 1 output\n" + "content line\n" * 20, encoding="utf-8")
        (self.proj / ".claude" / "state.yaml").write_text(
            "workflow:\n"
            "  name: wf\n  status: running\n  current_step: 1\n"
            "  autopilot:\n    enabled: true\n"
            "  outputs:\n    step-1: out-step-1.md\n",
            encoding="utf-8")
        md = snap.generate_snapshot_md("s1", "Stop", str(self.proj), self._entries())
        self.assertIn("단계별 산출물 검증", md)
        self.assertTrue("[OK]" in md or "[FAIL]" in md,
                        "validate_step_output must run and emit a per-step mark")

    def test_quality_gate_section_runs_parse_review_verdict(self):
        # REGRESSION: a review log makes generate_snapshot_md call
        # _extract_quality_gate_state → parse_review_verdict. Missing import
        # drops the entire 'Quality Gate State' section (swallowed by try/except).
        (self.proj / "review-logs").mkdir()
        (self.proj / "review-logs" / "step-2-review.md").write_text(
            "# Review — Step 2\n\n## Verdict\n\nPASS\n", encoding="utf-8")
        md = snap.generate_snapshot_md("s1", "Stop", str(self.proj), self._entries())
        self.assertIn("품질 게이트 상태", md)


class TestSnapshotHelpers(unittest.TestCase):
    def setUp(self):
        self.proj = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.proj)

    def test_is_rich_snapshot(self):
        self.assertFalse(snap.is_rich_snapshot("short poor snapshot"))
        rich = "\n".join([snap.E5_RICH_CONTENT_MARKER,
                          snap.E5_COMPLETION_STATE_MARKER,
                          snap.E5_DESIGN_DECISIONS_MARKER]) + "\n" + ("x" * 600)
        self.assertTrue(snap.is_rich_snapshot(rich))

    def test_get_snapshot_dir(self):
        d = snap.get_snapshot_dir(str(self.proj))
        self.assertIn("context-snapshots", d)

    def test_cleanup_snapshots_no_dir_is_safe(self):
        # Non-existent snapshot dir must not raise.
        snap.cleanup_snapshots(str(self.proj / "context-snapshots"))


if __name__ == "__main__":
    unittest.main()
