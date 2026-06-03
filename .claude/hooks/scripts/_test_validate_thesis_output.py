#!/usr/bin/env python3
"""Tests for validate_thesis_output.py — Thesis output structural validation (TO1-TO3).

Run: python3 -m pytest _test_validate_thesis_output.py -v
  or: python3 _test_validate_thesis_output.py
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import validate_thesis_output as vto


class TestThesisOutputValidation(unittest.TestCase):
    """Test thesis output structural validation rules TO1-TO3."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        # Wave 5 expects a single output (15-plagiarism-report.md, prefix PC),
        # which keeps the per-wave validation fixture minimal.
        self.wave_dir = self.tmpdir / "wave-results" / "wave-5"
        self.wave_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_report(self, content):
        path = self.wave_dir / "15-plagiarism-report.md"
        path.write_text(content, encoding="utf-8")
        return path

    def test_valid_output(self):
        content = (
            "# Plagiarism Report\n\n"
            "## Findings\n\n"
            "```yaml\n"
            "- claim_id: PC-001\n"
            "  type: EMPIRICAL\n"
            "  statement: No significant textual overlap detected in the corpus.\n"
            "  source: Turnitin 2024\n"
            "```\n\n"
            "Detailed similarity analysis confirms originality across all chapters, "
            "with the highest single-source match well below the accepted threshold.\n"
        )
        self._write_report(content)
        result = vto.validate_wave(str(self.tmpdir), 5)
        self.assertTrue(result["passed"], f"Valid wave should pass: {result}")

    def test_missing_file(self):
        # No output file written → TO1 missing-file error.
        result = vto.validate_wave(str(self.tmpdir), 5)
        self.assertFalse(result["passed"])

    def test_empty_file(self):
        self._write_report("")
        result = vto.validate_wave(str(self.tmpdir), 5)
        self.assertFalse(result["passed"])


class TestNoSystemSOTReference(unittest.TestCase):
    def test_no_state_yaml_reference(self):
        src = Path(__file__).parent / "validate_thesis_output.py"
        content = src.read_text(encoding="utf-8")
        self.assertNotIn("state.yaml", content,
                         "validate_thesis_output.py must not reference system SOT")


if __name__ == "__main__":
    unittest.main()
