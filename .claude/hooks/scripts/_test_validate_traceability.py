#!/usr/bin/env python3
"""Tests for validate_traceability.py — Cross-Step Traceability validation (CT1-CT5).

Run: python3 -m pytest _test_validate_traceability.py -v
  or: python3 _test_validate_traceability.py
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import validate_traceability as vt


class TestTraceabilityValidation(unittest.TestCase):
    """Test Cross-Step Traceability validation rules CT1-CT5."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_output(self, filename, content):
        path = self.tmpdir / filename
        path.write_text(content, encoding="utf-8")
        return str(path)

    def test_valid_traceability_markers(self):
        # CT2 requires each referenced step to have an output in SOT that exists.
        self._write_output("step-1.md", "# Step 1\n\n## Methodology\n\nDetails.\n")
        self._write_output("step-2.md", "# Step 2\n\n## Framework\n\nDetails.\n")
        self._write_output("step-3.md", "# Step 3\n\n## Results\n\nDetails.\n")
        content = (
            "# Analysis Output\n\n"
            "Based on previous findings [trace:step-1:methodology], "
            "we extend the framework [trace:step-2:framework]. "
            "The data supports [trace:step-3:results].\n"
        )
        self._write_output("step-5.md", content)
        sot = {"outputs": {
            "step-1": "step-1.md", "step-2": "step-2.md",
            "step-3": "step-3.md", "step-5": "step-5.md",
        }}
        is_valid, warnings = vt.validate_cross_step_traceability(
            str(self.tmpdir), 5, sot_data=sot)
        self.assertTrue(is_valid, f"Valid traceability should pass: {warnings}")

    def test_missing_file(self):
        sot = {"outputs": {"step-5": "nonexistent.md"}}
        is_valid, warnings = vt.validate_cross_step_traceability(
            str(self.tmpdir), 5, sot_data=sot)
        self.assertFalse(is_valid)

    def test_no_markers(self):
        self._write_output("step-5.md", "# Analysis Output\n\nNo cross-references here.\n")
        sot = {"outputs": {"step-5": "step-5.md"}}
        is_valid, warnings = vt.validate_cross_step_traceability(
            str(self.tmpdir), 5, sot_data=sot)
        # Should fail CT1 — no [trace:step-N:...] markers present
        self.assertFalse(is_valid)


class TestNoSystemSOTReference(unittest.TestCase):
    def test_no_state_yaml_reference(self):
        src = Path(__file__).parent / "validate_traceability.py"
        content = src.read_text(encoding="utf-8")
        self.assertNotIn("state.yaml", content,
                         "validate_traceability.py must not reference system SOT")


if __name__ == "__main__":
    unittest.main()
