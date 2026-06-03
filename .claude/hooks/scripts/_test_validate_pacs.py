#!/usr/bin/env python3
"""Tests for validate_pacs.py — pACS log validation (PA1-PA7).

Run: python3 -m pytest _test_validate_pacs.py -v
  or: python3 _test_validate_pacs.py
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import validate_pacs as vp


class TestPacsValidation(unittest.TestCase):
    """Test pACS log validation rules PA1-PA7."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.pacs_dir = self.tmpdir / "pacs-logs"
        self.pacs_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_pacs_log(self, step, content):
        path = self.pacs_dir / f"step-{step}-pacs.md"
        path.write_text(content, encoding="utf-8")
        return path

    def _make_valid_pacs(self, step=1, f_score=75, c_score=80, l_score=70):
        return (
            f"# pACS Log — Step {step}\n\n"
            f"## Pre-mortem Protocol\n\n"
            f"1. What could go wrong? The analysis might miss key papers.\n"
            f"2. What assumptions are weak? Database coverage assumption.\n"
            f"3. What would a critic say? Sample size too small.\n\n"
            f"## Dimensions\n\n"
            f"| Dimension | Score |\n"
            f"|-----------|-------|\n"
            f"| F (Faithfulness) | {f_score} |\n"
            f"| C (Completeness) | {c_score} |\n"
            f"| L (Logical Coherence) | {l_score} |\n\n"
            f"## pACS Score\n\n"
            f"pACS = min(F, C, L) = {min(f_score, c_score, l_score)}\n"
        )

    def test_valid_pacs_log(self):
        self._write_pacs_log(1, self._make_valid_pacs())
        is_valid, warnings = vp.validate_pacs_output(str(self.tmpdir), 1)
        self.assertTrue(is_valid, f"Valid pACS log should pass: {warnings}")

    def test_missing_file(self):
        is_valid, warnings = vp.validate_pacs_output(str(self.tmpdir), 99)
        self.assertFalse(is_valid)

    def test_empty_file(self):
        self._write_pacs_log(1, "")
        is_valid, warnings = vp.validate_pacs_output(str(self.tmpdir), 1)
        self.assertFalse(is_valid)


class TestNoSystemSOTReference(unittest.TestCase):
    def test_no_state_yaml_reference(self):
        src = Path(__file__).parent / "validate_pacs.py"
        content = src.read_text(encoding="utf-8")
        self.assertNotIn("state.yaml", content,
                         "validate_pacs.py must not reference system SOT")


if __name__ == "__main__":
    unittest.main()
