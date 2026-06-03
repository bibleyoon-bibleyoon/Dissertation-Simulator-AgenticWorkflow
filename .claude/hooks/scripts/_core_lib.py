#!/usr/bin/env python3
"""Foundation primitives shared across the Context Preservation System.

SOT path resolution, SOT schema validation, remediation extraction, and core
constants. Extracted from _context_lib.py per ADR-076 (Increment 1).
Has NO dependency on other _*_lib modules (DAG root).
"""

import json
import os
import re
import sys
import time
import fcntl
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


MIN_OUTPUT_SIZE = 100


SOT_FILENAMES = ("state.yaml", "state.yml", "state.json")


def sot_paths(project_dir):
    """Build SOT file path list from SOT_FILENAMES constant (A-3: single definition)."""
    return [os.path.join(project_dir, ".claude", fn) for fn in SOT_FILENAMES]


def validate_sot_schema(ap_state):
    """SOT Schema Validation: structural integrity of autopilot state dict.

    P1 Compliance: All checks are deterministic (type, range, format).
    SOT Compliance: Read-only — validates in-memory dict, no file I/O.
    No duplication: file existence is validate_step_output()'s responsibility.

    Args:
        ap_state: dict from read_autopilot_state(), or None

    Returns: list of warning strings (empty list = all checks passed)
    """
    if not ap_state or not isinstance(ap_state, dict):
        return []

    warnings = []

    # S1: current_step — must be int >= 0
    cs = ap_state.get("current_step")
    if cs is not None:
        if not isinstance(cs, int):
            warnings.append(
                f"SOT schema: current_step is {type(cs).__name__}, expected int"
            )
        elif cs < 0:
            warnings.append(f"SOT schema: current_step is {cs}, must be >= 0")

    # S2: outputs — must be dict
    outputs = ap_state.get("outputs")
    if outputs is not None and not isinstance(outputs, dict):
        warnings.append(
            f"SOT schema: outputs is {type(outputs).__name__}, expected dict"
        )

    # S3: outputs keys — must follow step-N or step-N-ko format
    if isinstance(outputs, dict):
        for key in outputs:
            if not isinstance(key, str) or not key.startswith("step-"):
                warnings.append(f"SOT schema: invalid output key '{key}'")
                continue
            # Extract step number — allow step-N and step-N-ko (translation)
            suffix = key[5:]  # after "step-"
            parts = suffix.split("-", 1)
            if not parts[0].isdigit():
                warnings.append(
                    f"SOT schema: output key '{key}' has non-numeric step number"
                )

    # S4: No output recorded for future steps (step number > current_step)
    if isinstance(cs, int) and isinstance(outputs, dict):
        for key in outputs:
            if isinstance(key, str) and key.startswith("step-"):
                suffix = key[5:]
                parts = suffix.split("-", 1)
                if parts[0].isdigit():
                    step_num = int(parts[0])
                    if step_num > cs:
                        warnings.append(
                            f"SOT schema: output '{key}' for future step "
                            f"(current_step={cs})"
                        )

    # S5: workflow_status — must be recognized value
    status = ap_state.get("workflow_status", "")
    if status:
        valid_statuses = {"running", "completed", "error", "paused"}
        if status not in valid_statuses:
            warnings.append(
                f"SOT schema: unrecognized workflow_status '{status}'"
            )

    # S6: auto_approved_steps — items must be int, within plausible range
    approved = ap_state.get("auto_approved_steps", [])
    if isinstance(approved, list):
        for item in approved:
            if not isinstance(item, int):
                warnings.append(
                    f"SOT schema: auto_approved_steps contains non-int: {item}"
                )
            elif isinstance(cs, int) and item > cs:
                warnings.append(
                    f"SOT schema: auto_approved_steps contains future step "
                    f"{item} (current_step={cs})"
                )

    # S7: pacs — must be dict with valid structure (if present)
    pacs = ap_state.get("pacs")
    if pacs is not None:
        if not isinstance(pacs, dict):
            warnings.append(
                f"SOT schema: pacs is {type(pacs).__name__}, expected dict"
            )
        else:
            # S7a: dimensions — dict with F, C, L keys (int 0-100)
            dims = pacs.get("dimensions")
            if dims is not None:
                if not isinstance(dims, dict):
                    warnings.append("SOT schema: pacs.dimensions must be dict")
                else:
                    for dim_key in ("F", "C", "L"):
                        dim_val = dims.get(dim_key)
                        if dim_val is not None:
                            if not isinstance(dim_val, (int, float)):
                                warnings.append(
                                    f"SOT schema: pacs.dimensions.{dim_key} is "
                                    f"{type(dim_val).__name__}, expected int"
                                )
                            elif not (0 <= dim_val <= 100):
                                warnings.append(
                                    f"SOT schema: pacs.dimensions.{dim_key} = "
                                    f"{dim_val}, must be 0-100"
                                )
            # S7b: current_step_score — int 0-100
            score = pacs.get("current_step_score")
            if score is not None:
                if not isinstance(score, (int, float)):
                    warnings.append(
                        f"SOT schema: pacs.current_step_score is "
                        f"{type(score).__name__}, expected int"
                    )
                elif not (0 <= score <= 100):
                    warnings.append(
                        f"SOT schema: pacs.current_step_score = {score}, "
                        f"must be 0-100"
                    )
            # S7c: weak_dimension — must be one of F, C, L
            weak = pacs.get("weak_dimension")
            if weak is not None and weak not in ("F", "C", "L"):
                warnings.append(
                    f"SOT schema: pacs.weak_dimension = '{weak}', "
                    f"must be one of F, C, L"
                )
            # S7d: history — must be dict of step-keys → {score, weak}
            # Schema: claude-code-patterns.md §SOT pacs 필드 스키마
            #   history:
            #     step-1: {score: 85, weak: "C"}
            history = pacs.get("history")
            if history is not None:
                if not isinstance(history, dict):
                    warnings.append(
                        f"SOT schema: pacs.history is "
                        f"{type(history).__name__}, expected dict"
                    )
                else:
                    for hkey, hval in history.items():
                        if not isinstance(hval, dict):
                            warnings.append(
                                f"SOT schema: pacs.history.{hkey} is "
                                f"{type(hval).__name__}, expected dict"
                            )
                            continue
                        hscore = hval.get("score")
                        if hscore is not None:
                            if not isinstance(hscore, (int, float)):
                                warnings.append(
                                    f"SOT schema: pacs.history.{hkey}.score "
                                    f"is {type(hscore).__name__}, expected int"
                                )
                            elif not (0 <= hscore <= 100):
                                warnings.append(
                                    f"SOT schema: pacs.history.{hkey}.score "
                                    f"= {hscore}, must be 0-100"
                                )
                        hweak = hval.get("weak")
                        if hweak is not None and hweak not in ("F", "C", "L"):
                            warnings.append(
                                f"SOT schema: pacs.history.{hkey}.weak "
                                f"= '{hweak}', must be F, C, or L"
                            )
            # S7e: pre_mortem_flag — must be string (if present)
            pmf = pacs.get("pre_mortem_flag")
            if pmf is not None and not isinstance(pmf, str):
                warnings.append(
                    f"SOT schema: pacs.pre_mortem_flag is "
                    f"{type(pmf).__name__}, expected string"
                )

    # S8: active_team — must be dict with required fields (if present)
    active_team = ap_state.get("active_team")
    if active_team is not None:
        if not isinstance(active_team, dict):
            warnings.append(
                f"SOT schema: active_team is {type(active_team).__name__}, "
                f"expected dict"
            )
        else:
            # S8a: name — must be non-empty string
            team_name = active_team.get("name")
            if team_name is not None and not isinstance(team_name, str):
                warnings.append("SOT schema: active_team.name must be string")
            # S8b: status — must be recognized value
            # Schema: claude-code-patterns.md §SOT 갱신 프로토콜
            #   "partial" (팀 작업 진행 중) | "all_completed" (모든 Task 완료)
            team_status = active_team.get("status")
            valid_team_statuses = {"partial", "all_completed"}
            if team_status and team_status not in valid_team_statuses:
                warnings.append(
                    f"SOT schema: active_team.status '{team_status}' "
                    f"unrecognized (expected: partial | all_completed)"
                )
            # S8c: tasks_completed — must be list (if present)
            tc = active_team.get("tasks_completed")
            if tc is not None and not isinstance(tc, list):
                warnings.append(
                    f"SOT schema: active_team.tasks_completed is "
                    f"{type(tc).__name__}, expected list"
                )
            # S8d: tasks_pending — must be list (if present)
            tp = active_team.get("tasks_pending")
            if tp is not None and not isinstance(tp, list):
                warnings.append(
                    f"SOT schema: active_team.tasks_pending is "
                    f"{type(tp).__name__}, expected list"
                )
            # S8e: completed_summaries — must be dict (if present)
            cs_summaries = active_team.get("completed_summaries")
            if cs_summaries is not None:
                if not isinstance(cs_summaries, dict):
                    warnings.append(
                        f"SOT schema: active_team.completed_summaries is "
                        f"{type(cs_summaries).__name__}, expected dict"
                    )
                else:
                    for task_id, info in cs_summaries.items():
                        if not isinstance(info, dict):
                            warnings.append(
                                f"SOT schema: active_team.completed_summaries"
                                f".{task_id} must be dict"
                            )

    # S9: outputs key suffix must be a recognized language code (en | ko) or absent
    # Stricter enforcement than S3: prevents agents from writing arbitrary suffix keys
    VALID_OUTPUT_LANG_SUFFIXES = frozenset({"en", "ko"})
    if isinstance(outputs, dict):
        for key in outputs:
            if not isinstance(key, str) or not key.startswith("step-"):
                continue  # Already reported in S3
            suffix = key[5:]  # after "step-"
            parts = suffix.split("-", 1)
            if parts[0].isdigit() and len(parts) > 1:
                lang_suffix = parts[1]
                if lang_suffix not in VALID_OUTPUT_LANG_SUFFIXES:
                    warnings.append(
                        f"SOT schema: output key '{key}' has unrecognized language suffix "
                        f"'{lang_suffix}' (expected: en | ko) — possible agent hallucination"
                    )

    # S10: pacs.history step numbers must not exceed current_step
    # Prevents agents from recording PACS scores for future steps they haven't executed
    # Note: pacs already bound at S7 (line 726) — no re-read needed
    if isinstance(pacs, dict) and isinstance(cs, int):
        history = pacs.get("history")
        if isinstance(history, dict):
            for hkey in history:
                if isinstance(hkey, str) and hkey.startswith("step-"):
                    step_num_str = hkey[5:]
                    if step_num_str.isdigit():
                        step_num = int(step_num_str)
                        if step_num > cs:
                            warnings.append(
                                f"SOT schema: pacs.history '{hkey}' records data "
                                f"for future step (current_step={cs}) — "
                                f"possible parallel agent data inconsistency"
                            )

    # S11: pccs schema validation (if present)
    # Validates pccs SOT block structure when pCCS is active.
    # pccs is optional — only validated when the key exists.
    pccs = ap_state.get("pccs")
    if pccs is not None:
        if not isinstance(pccs, dict):
            warnings.append("SOT schema: 'pccs' must be a dict")
        else:
            # S11a: cal_delta must be numeric
            cal = pccs.get("cal_delta")
            if cal is not None and not isinstance(cal, (int, float)):
                warnings.append(f"SOT schema: pccs.cal_delta must be numeric, got {type(cal).__name__}")
            # S11b: last_step must not exceed current_step
            last_step = pccs.get("last_step")
            if isinstance(last_step, int) and isinstance(cs, int) and last_step > cs:
                warnings.append(
                    f"SOT schema: pccs.last_step={last_step} exceeds "
                    f"current_step={cs} — possible future data"
                )
            # S11c: history entries must have required fields
            pccs_hist = pccs.get("history")
            if isinstance(pccs_hist, dict):
                for pkey, pval in pccs_hist.items():
                    if isinstance(pval, dict):
                        if "mean_pccs" not in pval:
                            warnings.append(f"SOT schema: pccs.history.{pkey} missing 'mean_pccs'")
                        if "action" not in pval:
                            warnings.append(f"SOT schema: pccs.history.{pkey} missing 'action'")
                        # S11d: mode must be FULL or DEGRADED if present
                        mode = pval.get("mode")
                        if mode is not None and mode not in ("FULL", "DEGRADED"):
                            warnings.append(f"SOT schema: pccs.history.{pkey}.mode='{mode}' invalid (expected FULL/DEGRADED)")

    return warnings


def extract_remediations(warnings, remediations_dict):
    """Central remediation extraction — replaces 7 inline loops across validators.

    P1 Compliance: Deterministic prefix matching + completeness self-check.
    Called by all validate_*.py scripts after validation completes.

    Matches warnings starting with "{CODE} FAIL" to _REMEDIATIONS keys.
    Also performs P1-F completeness self-check: if a FAIL code has no matching
    remediation entry, adds a warning to the returned dict.

    Args:
        warnings: list of warning strings from validator (e.g., ["PA1 FAIL: ..."])
        remediations_dict: dict mapping check codes to fix instructions

    Returns:
        dict: {code: remediation_text, ...} for matched FAIL codes.
              If a FAIL code has no remediation, includes:
              {code: "NO_REMEDIATION: check code '{code}' missing from _REMEDIATIONS"}
    """
    if not warnings or not remediations_dict:
        return {}

    result = {}
    for w in warnings:
        # C-1: Defensive type guard — non-string elements (None, int, etc.) skip
        if not isinstance(w, str) or "FAIL" not in w:
            continue
        matched = False
        for code in remediations_dict:
            if w.startswith(f"{code} FAIL"):
                result[code] = remediations_dict[code]
                matched = True
                break
        # P1-F: Completeness self-check — detect missing remediation entries
        if not matched:
            # Extract the check code prefix (e.g., "PA1" from "PA1 FAIL: ...")
            fail_idx = w.find(" FAIL")
            if fail_idx > 0:
                fail_code = w[:fail_idx].strip()
                # H-2: Only accept valid code format (PA1, T9, V1a, RV1, etc.)
                # Reject multi-word or malformed prefixes
                if (fail_code
                        and re.match(r'^[A-Z]+\d+[a-z]?$', fail_code)
                        and fail_code not in result):
                    result[fail_code] = (
                        f"NO_REMEDIATION: check code '{fail_code}' missing "
                        f"from _REMEDIATIONS — add an entry to the validator script"
                    )
    return result


# --- Truncation limits + tool-error patterns + _truncate helper -------------
# Moved from _context_lib.py per ADR-077 (Increment 2); cross-cutting foundation
# used by capture, snapshot, and diagnosis. (ADR-030 constant centralization.)
EDIT_PREVIEW_CHARS = 1000


ERROR_RESULT_CHARS = 3000


NORMAL_RESULT_CHARS = 1500


WRITE_PREVIEW_CHARS = 500


GENERIC_INPUT_CHARS = 200


BASH_CMD_CHARS = 200


TASK_PROMPT_CHARS = 200


SOT_CAPTURE_CHARS = 3000


TOOL_ERROR_PATTERNS = [
    "Error:", "error:", "FAILED", "failed",
    "not found", "Permission denied", "No such file",
]


def _truncate(text, max_len):
    """Truncate text to max_len, adding ellipsis if needed."""
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."

