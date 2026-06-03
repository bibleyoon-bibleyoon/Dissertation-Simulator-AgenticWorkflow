#!/usr/bin/env python3
"""
Context Preservation System — Shared Library (_context_lib.py)

All hook scripts share this module for:
- Transcript JSONL parsing with deterministic extraction rules
- Structured MD snapshot generation (facts only, no heuristic inference)
- SOT state capture
- Atomic file writes with locking
- Token estimation (multi-signal)
- Dedup guard

Architecture:
  RLM Pattern: Snapshots are external memory objects (files on disk).
  P1 Compliance: Code handles deterministic extraction only.
                  Semantic interpretation is Claude's responsibility.
  SOT Compliance: Read-only access to SOT; writes only to context-snapshots/.
  Quality First: 100% accurate structured data, zero heuristic inference.
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

# Ensure sibling modules (_core_lib, _validation_lib) resolve even when this
# module is loaded by file path without the scripts dir on sys.path (ADR-076).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- ADR-076 Increment 1: re-export shim ---------------------------------
# Foundation primitives and quality-gate validators were extracted to
# _core_lib.py and _validation_lib.py. These star-imports preserve
# _context_lib's public interface so the ~40 importers stay unchanged.
from _core_lib import *  # noqa: F401,F403
from _validation_lib import *  # noqa: F401,F403
from _capture_lib import *  # noqa: F401,F403
# Underscore-prefixed symbols imported by external scripts are not covered
# by `import *`, so re-export them explicitly:
from _validation_lib import (  # noqa: F401
    _DKS_REF_RE,
    _find_translation_files_for_step,
    _TRACE_MARKER_RE,
)
from _core_lib import _truncate  # underscore; used by remaining snapshot/diagnosis


# =============================================================================
# Constants
# =============================================================================

# Token estimation: mixed Korean/English content ≈ 2.5 chars/token
CHARS_PER_TOKEN = 2.5
# Claude's context window (200K tokens)
CONTEXT_WINDOW_TOKENS = 200_000
# System prompt overhead estimate (tokens)
SYSTEM_OVERHEAD_TOKENS = 15_000
# Effective capacity
EFFECTIVE_CAPACITY = CONTEXT_WINDOW_TOKENS - SYSTEM_OVERHEAD_TOKENS
# 75% threshold
THRESHOLD_75_TOKENS = int(EFFECTIVE_CAPACITY * 0.75)
# Snapshot size target (characters) — Quality First (절대 기준 1)
# Read tool handles up to 2000 lines. 100KB preserves decision context.
MAX_SNAPSHOT_CHARS = 100_000
# Dedup guard window (seconds) — reduced to avoid missing rapid changes
DEDUP_WINDOW_SECONDS = 5
# Stop hook uses wider window to reduce noise (~60→~10 saves/hour)
STOP_DEDUP_WINDOW_SECONDS = 30
# Max snapshots to retain per trigger type
MAX_SNAPSHOTS = {
    "precompact": 3,
    "sessionend": 3,
    "threshold": 2,
    "stop": 5,
}
DEFAULT_MAX_SNAPSHOTS = 3
# Knowledge Archive limits (Area 1: Cross-Session Knowledge Archive)
MAX_KNOWLEDGE_INDEX_ENTRIES = 200
MAX_SESSION_ARCHIVES = 20
# E5 Empty Snapshot Guard — section header constants
# These constants are the single definition for section headers used in both
# generate_snapshot_md() and is_rich_snapshot(). Changing a constant here
# automatically updates both the snapshot generator and the E5 Guard detector.
E5_RICH_CONTENT_MARKER = "### 수정 중이던 파일"
E5_COMPLETION_STATE_MARKER = "## 결정론적 완료 상태"
E5_DESIGN_DECISIONS_MARKER = "## 주요 설계 결정"
# A1: Multi-signal rich content markers for E5 Guard
# is_rich_snapshot() checks `marker in content` (substring match).
# Full headers in snapshot include English suffix, e.g.:
#   "## 결정론적 완료 상태 (Deterministic Completion State)"
E5_RICH_SIGNALS = [
    E5_RICH_CONTENT_MARKER,         # "### 수정 중이던 파일"
    E5_COMPLETION_STATE_MARKER,     # "## 결정론적 완료 상태"
    E5_DESIGN_DECISIONS_MARKER,     # "## 주요 설계 결정"
]

# P1-RLM: Snapshot Section Markers — enables selective peek (programmatic access)
# HTML comments are invisible to normal readers but parseable by restore_context.py.
# Keys are stable identifiers; values are the exact marker strings.
# restore_context.py imports this dict to avoid constant duplication (shotgun surgery prevention).
SNAPSHOT_SECTION_MARKERS = {
    "header":           "<!-- SECTION:header -->",
    "task":             "<!-- SECTION:task -->",
    "next_step":        "<!-- SECTION:next_step -->",
    "sot":              "<!-- SECTION:sot -->",
    "autopilot":        "<!-- SECTION:autopilot -->",
    "quality_gate":     "<!-- SECTION:quality_gate -->",
    "team":             "<!-- SECTION:team -->",
    "ulw":              "<!-- SECTION:ulw -->",
    "diagnosis":        "<!-- SECTION:diagnosis -->",
    "decisions":        "<!-- SECTION:decisions -->",
    "resume":           "<!-- SECTION:resume -->",
    "completion":       "<!-- SECTION:completion -->",
    "git":              "<!-- SECTION:git -->",
    "modified_files":   "<!-- SECTION:modified_files -->",
    "referenced_files": "<!-- SECTION:referenced_files -->",
    "user_messages":    "<!-- SECTION:user_messages -->",
    "responses":        "<!-- SECTION:responses -->",
    "statistics":       "<!-- SECTION:statistics -->",
    "commands":         "<!-- SECTION:commands -->",
    "work_log":         "<!-- SECTION:work_log -->",
}

# --- Path tag extraction constants (A3: language-independent search tags) ---
_PATH_SKIP_NAMES = frozenset({
    "src", "lib", "dist", "build", "node_modules", "venv", ".git",
    "tests", "test", "__pycache__", ".claude", "scripts", "hooks",
})
_EXT_TAGS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "react", ".jsx": "react", ".md": "markdown",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json",
    ".sh": "shell", ".css": "css", ".html": "html",
    ".rs": "rust", ".go": "golang", ".java": "java",
}

# --- Predictive Debugging: Risk Score Constants (P1 — module-level) ---
# Used by aggregate_risk_scores() and validate_risk_scores()
# Weights per error type: higher = more indicative of fragile code
# D-7: Keys MUST match ERROR_TAXONOMY in _classify_error_patterns() (~line 2812)
#      + "unknown" for unclassified errors. Mismatch → fallback weight 0.7 applied.
_RISK_WEIGHTS = {
    "edit_mismatch": 2.0,   # File structure instability (frequent edit failures)
    "dependency": 2.5,       # High ripple effect
    "type_error": 1.5,       # Type complexity
    "syntax": 1.0,           # Repetitive — complex file indicator
    "value_error": 1.0,
    "git_error": 1.0,
    "timeout": 0.5,          # Often environmental, not code
    "file_not_found": 0.5,   # Usually one-time
    "permission": 0.5,
    "connection": 0.3,       # Network — may not be code issue
    "memory": 0.3,
    "command_not_found": 0.3,
    "unknown": 0.7,          # ~30% of errors — ignoring loses significant data
}
# Recency decay: (max_days, weight_multiplier)
# More recent errors are more relevant to current code state
_RECENCY_DECAY_DAYS = [
    (30, 1.0),              # 0-30 days: full weight
    (90, 0.5),              # 31-90 days: half weight
    (float("inf"), 0.25),   # 91+ days: quarter weight
]
# Minimum risk score to trigger PreToolUse warning
_RISK_SCORE_THRESHOLD = 3.0
# Minimum sessions in knowledge-index before activation (cold start guard)
_RISK_MIN_SESSIONS = 5

# --- Pre-compiled regex patterns (module-level — compiled once per process) ---
# Used by _extract_next_step()
_NEXT_STEP_RE = re.compile(
    r'(?:다음으로|이제|그 다음|그 후|Next,?|Now |Then )'
    r'\s*(.{10,500}?)(?:\.\s|\n\n|$)',
    re.MULTILINE,
)
# Used by _extract_decisions()
_DECISION_MARKER_RE = re.compile(r'<!--\s*DECISION:\s*(.+?)\s*-->', re.DOTALL)
_DECISION_BOLD_RE = re.compile(
    r'\*\*(?:Decision|결정|선택|채택|판단)\s*(?::|：)\*\*\s*(.+?)(?:\n|$)',
    re.IGNORECASE,
)
_DECISION_INTENT_NOISE_RE = re.compile(
    r'읽겠습니다|확인하겠습니다|시작하겠습니다|살펴보겠습니다|'
    r'진행하겠습니다|분석하겠습니다|검토하겠습니다|파악하겠습니다|'
    r'Let me read|Let me check|I\'ll start|I\'ll look',
    re.IGNORECASE,
)
_DECISION_INTENT_RE = re.compile(
    r'(?:^|\n)\s*[-*]?\s*(.{10,120}?(?:하겠습니다|로 결정|을 선택|를 채택|접근 방식|approach))',
    re.MULTILINE,
)
_DECISION_RATIONALE_RE = re.compile(
    r'(?:선택\s*이유|근거|Rationale|Reason(?:ing)?)\s*(?::|：)\s*(.+?)(?:\n|$)',
    re.IGNORECASE,
)
_DECISION_COMPARISON_RE = re.compile(
    r'(.{5,80}?)\s+(?:대신|보다는?|rather than|instead of|over)\s+(.{5,80}?)(?:\.|,|\n|$)',
    re.IGNORECASE | re.MULTILINE,
)
_DECISION_TRADEOFF_RE = re.compile(
    r'(?:trade-?off|장단점|pros?\s*(?:and|&)\s*cons?|단점은|downside)\s*(?::|：|은|는)?\s*(.+?)(?:\n|$)',
    re.IGNORECASE,
)
_DECISION_CHOICE_RE = re.compile(
    r'(?:chose|opted for|selected|decided to|went with|picked)\s+(.{10,150}?)(?:\.|,|\n|$)',
    re.IGNORECASE,
)
# Used by generate_snapshot_md() — system command filter for "현재 작업" section
_SYSTEM_CMD_RE = re.compile(
    r'^\s*<command-name>|^\s*/(?:clear|help|compact|init|resume|review|login|logout|mcp|config)\b',
    re.IGNORECASE | re.MULTILINE,
)

# Used by Abductive Diagnosis functions — diagnosis-logs/ parsing
# Captures the FULL heading line (including H-ID) so AD9 can extract H[1-4] from it.
# E.g., "## H1: Upstream data quality issue" → captures "H1: Upstream data quality issue"
_DIAG_HYPOTHESIS_RE = re.compile(
    r"^#+\s*((?:H\d|Hypothesis)\b.+)", re.MULTILINE | re.IGNORECASE,
)
_DIAG_SELECTED_RE = re.compile(
    r"(?:Selected|Chosen|Primary)\s*(?:Hypothesis|H\d)\s*:\s*(.+)",
    re.IGNORECASE,
)
_DIAG_EVIDENCE_RE = re.compile(
    r"^-\s*\*?\*?Evidence\*?\*?\s*:\s*(.+)", re.MULTILINE | re.IGNORECASE,
)
_DIAG_GATE_RE = re.compile(
    r"Gate\s*:\s*(verification|pacs|review)", re.IGNORECASE,
)
_DIAG_SOURCE_STEP_RE = re.compile(
    r"\(source:\s*Step\s+(\d+)\)", re.IGNORECASE,
)


# =============================================================================
# Per-File Diff Stats (C-4)
# =============================================================================

def _get_per_file_diff_stats(project_dir):
    """Get per-file line change counts from git diff --numstat.

    P1 Compliance: deterministic subprocess output.
    Returns: dict of {filepath: (added, removed)} or empty dict.
    """
    try:
        proc = subprocess.run(
            ["git", "diff", "--numstat", "HEAD"],
            cwd=project_dir, capture_output=True, text=True, timeout=5
        )
        if proc.returncode != 0:
            return {}
        result = {}
        for line in proc.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                added, removed, filepath = parts[0], parts[1], parts[2]
                result[filepath] = (added, removed)
        return result
    except Exception:
        return {}


# =============================================================================
# Next Step Extraction (CM-3)
# =============================================================================

def _extract_next_step(assistant_texts):
    """CM-3: Extract forward-looking statement from last assistant response.

    Captures the next action Claude was about to take, enabling task-based
    session resumption instead of summary-based guessing.

    P1 Compliance: Regex-based deterministic extraction.
    Returns: str or None (first match from last response, max 500 chars).
    """
    if not assistant_texts:
        return None

    # FIX-M4: Search last 5 assistant responses (expanded from 3) for forward-looking patterns
    # In long sessions (100+ turns), actual next-step may not be in last 3 responses
    # CM-F: Expanded from 200→500 chars to preserve structured action plans
    # Pattern: module-level _NEXT_STEP_RE (compiled once per process)
    for entry in reversed(assistant_texts[-5:]):
        content = entry.get("content", "")
        match = _NEXT_STEP_RE.search(content)
        if match:
            return match.group(0).strip()[:500]
    return None


# =============================================================================
# Decision Extraction (C-1)
# =============================================================================

def _extract_decisions(assistant_texts):
    """Extract structured design decisions from assistant responses.

    Detects:
    1. Explicit markers: <!-- DECISION: ... -->
    2. Structured patterns: **Decision:** / **결정:** / **선택:**
    3. Implicit intent patterns: "~하겠습니다", "선택 이유:", "approach:"
    4. Rationale patterns: "이유:", "근거:", "Rationale:", "because"

    P1 Compliance: Regex-based deterministic extraction.
    Returns: list of decision strings (max 20).
    """
    decisions = []
    # All patterns are module-level pre-compiled (_DECISION_*_RE constants)
    # Pattern 1: HTML comment markers (_DECISION_MARKER_RE)
    # Pattern 2: Bold markers (_DECISION_BOLD_RE)
    # Pattern 3: Implicit intent + noise filter (_DECISION_INTENT_RE, _DECISION_INTENT_NOISE_RE)
    # Pattern 4: Rationale (_DECISION_RATIONALE_RE)
    # Pattern 5-7: Comparison, trade-off, choice (_DECISION_COMPARISON_RE, _DECISION_TRADEOFF_RE, _DECISION_CHOICE_RE)

    for entry in assistant_texts:
        content = entry.get("content", "")
        for match in _DECISION_MARKER_RE.finditer(content):
            decisions.append(("[explicit] " + match.group(1).strip())[:300])
        for match in _DECISION_BOLD_RE.finditer(content):
            decisions.append(("[decision] " + match.group(1).strip())[:300])
        for match in _DECISION_INTENT_RE.finditer(content):
            matched_text = match.group(1).strip()
            # CM-2: Skip routine action declarations (noise)
            if _DECISION_INTENT_NOISE_RE.search(matched_text):
                continue
            decisions.append(("[intent] " + matched_text)[:300])
        for match in _DECISION_RATIONALE_RE.finditer(content):
            decisions.append(("[rationale] " + match.group(1).strip())[:300])
        # CM-A + E-2: New high-signal decision patterns
        for match in _DECISION_COMPARISON_RE.finditer(content):
            decisions.append(("[decision] " + match.group(0).strip())[:300])
        for match in _DECISION_TRADEOFF_RE.finditer(content):
            decisions.append(("[rationale] " + match.group(0).strip())[:300])
        for match in _DECISION_CHOICE_RE.finditer(content):
            decisions.append(("[decision] " + match.group(0).strip())[:300])

    # Dedup while preserving order
    seen = set()
    unique = []
    for d in decisions:
        if d not in seen:
            seen.add(d)
            unique.append(d)

    # CM-2 + FIX-M1: Stratified slot allocation — 20 slots total
    # High-signal: [explicit] up to 5, [decision] up to 7, [rationale] up to 5
    # Overflow: [intent] fills remaining slots (noise-reduced via filter)
    _DECISION_PRIORITY = {"[explicit]": 0, "[decision]": 1, "[rationale]": 2, "[intent]": 3}
    # B-3: Safer tag extraction — use find() on prefix only to avoid false matches
    # from ']' characters in the decision content itself
    def _get_decision_tag(d):
        if d.startswith("["):
            end = d.find("]")
            if 0 < end < 20:  # Tags are short ([explicit], [intent], etc.)
                return d[:end + 1]
        return ""
    unique.sort(key=lambda d: _DECISION_PRIORITY.get(_get_decision_tag(d), 4))

    # FIX-M1: Expanded from 15→20 slots with proportional allocation
    # High-signal slots (up to 15): [explicit] + [decision] + [rationale]
    # Overflow slots (up to 5): [intent] fills remaining capacity
    high_signal = [d for d in unique if not d.startswith("[intent]")]
    intent_only = [d for d in unique if d.startswith("[intent]")]
    high_count = min(len(high_signal), 15)
    intent_budget = 20 - high_count  # Intent gets whatever high-signal doesn't use
    result = high_signal[:high_count] + intent_only[:intent_budget]
    return result[:20]


# =============================================================================
# MD Snapshot Generation (Deterministic Data Only)
# =============================================================================

def generate_snapshot_md(session_id, trigger, project_dir, entries, work_log=None, sot_content=None):
    """Generate comprehensive MD snapshot from parsed entries.

    Design Principle (P1 + RLM):
      - Code produces ONLY deterministic, structured facts
      - NO heuristic inference (progress, decisions, pending actions)
      - Claude interprets meaning when reading the snapshot

    v3 Enhancements:
      - E7: Deterministic Completion State (hallucination prevention)
      - E2: Git state capture (ground truth, post-commit aware)
      - E3: Per-edit detail preservation (aggregation loss prevention)
      - E4: Claude response priority selection + section promotion

    Section survival priority (truncation order):
      1-10: IMMORTAL  (Header, Task, Next Step*, SOT, Autopilot*, ULW*, Team*, Decisions*, Resume, Completion State, Git)
      11-14: CRITICAL  (Modified Files, Referenced Files, User Messages, Claude Responses)
      15-17: SACRIFICABLE (Statistics, Commands, Work Log)
      (* = conditional sections, only present when active)
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Classify entries
    user_messages = [e for e in entries if e["type"] == "user_message"]
    assistant_texts = [e for e in entries if e["type"] == "assistant_text"]
    tool_uses = [e for e in entries if e["type"] == "tool_use"]

    # Filtered user messages (exclude system-injected tags like <system-reminder>)
    user_msgs_filtered = [
        m for m in user_messages
        if not (m["content"].startswith("<") and ">" in m["content"][:50])
    ]

    # Pre-compute structured data (used by multiple sections)
    file_ops = _extract_file_operations(tool_uses, work_log)
    read_ops = _extract_read_operations(tool_uses)
    completion_state = extract_completion_state(entries, project_dir)
    git_state = capture_git_state(project_dir)
    conversation_phase = detect_conversation_phase(tool_uses)  # C-5
    phase_transitions = detect_phase_transitions(tool_uses)  # P1-3: multi-phase flow
    diff_stats = _get_per_file_diff_stats(project_dir)  # C-4
    decisions = _extract_decisions(assistant_texts)  # C-1

    # Build MD sections
    sections = []

    # ━━━ SURVIVAL PRIORITY 1: IMMORTAL ━━━
    SM = SNAPSHOT_SECTION_MARKERS  # shorthand

    # Header (P1-3: include phase flow if multi-phase detected)
    sections.append(SM["header"])
    sections.append(f"# Context Recovery — Session {session_id}")
    sections.append(f"> Saved: {now} | Trigger: {trigger}")
    sections.append(f"> Project: {project_dir}")
    sections.append(f"> Total entries: {len(entries)} | User msgs: {len(user_messages)} | Tool uses: {len(tool_uses)}")
    if len(phase_transitions) > 1:
        phase_flow = " → ".join(
            f"{t[0]}({t[2]-t[1]})" for t in phase_transitions
        )
        sections.append(f"> Phase flow: {phase_flow}")
    else:
        sections.append(f"> Phase: {conversation_phase}")
    sections.append("")

    # Section 1: Current Task (first + last user message — verbatim)
    # CM-6: IMMORTAL — user messages are the ground truth for task context
    sections.append(SM["task"])
    sections.append("## 현재 작업 (Current Task)")
    sections.append("<!-- IMMORTAL: 사용자 작업 지시 — 세션 복원의 핵심 맥락 -->")
    # CM-C: Filter system commands (/clear, /help, etc.) — show real task, not commands
    # Pattern: module-level _SYSTEM_CMD_RE (compiled once per process)
    real_user_msgs = [m for m in user_messages if not _SYSTEM_CMD_RE.search(m.get("content", ""))]
    if real_user_msgs:
        first_msg = real_user_msgs[0]["content"]
        sections.append(_truncate(first_msg, 3000))
        # Last instruction from filtered (non-continuation) messages
        real_filtered = [m for m in user_msgs_filtered if not _SYSTEM_CMD_RE.search(m.get("content", ""))]
        if real_filtered and len(real_filtered) > 1:
            last_msg = real_filtered[-1]["content"]
            if last_msg != first_msg:
                sections.append("")
                sections.append(f"**최근 지시 (Latest Instruction):** {_truncate(last_msg, 1500)}")
    elif user_messages:
        # Fallback: all messages are system commands, show the first one anyway
        sections.append(_truncate(user_messages[0]["content"], 3000))
    else:
        sections.append("(사용자 메시지 없음)")

    sections.append("")

    # Section 1.5: Next Step (IMMORTAL — cognitive resumption anchor)
    # CM-3: Promoted to independent IMMORTAL section for Phase 7 hard truncate survival
    next_step = _extract_next_step(assistant_texts)
    if next_step:
        sections.append(SM["next_step"])
        sections.append("## 다음 단계 (Next Step)")
        sections.append("<!-- IMMORTAL: 세션 복원 시 인지적 연속성의 핵심 — 다음 행동 지시 -->")
        sections.append(next_step)
        sections.append("")

    # Section 2: SOT State (deterministic file read)
    sections.append(SM["sot"])
    sections.append("## SOT 상태 (Workflow State)")
    sections.append("<!-- IMMORTAL: SOT 상태는 세션 복원 시 반드시 보존 — 워크플로우 진행 상태의 핵심 -->")
    if sot_content:
        sections.append(f"파일: `{sot_content['path']}`")
        sections.append(f"수정 시각: {sot_content['mtime']}")
        sections.append("```yaml")
        sections.append(sot_content["content"])
        sections.append("```")
    else:
        sections.append("SOT 파일 없음 (state.yaml/state.json 미발견)")
    sections.append("")

    # Section 2.5: Autopilot State (IMMORTAL — conditional, only when active)
    try:
        ap_state = read_autopilot_state(project_dir)
        if ap_state:
            sections.append(SM["autopilot"])
            sections.append("## Autopilot 상태 (Autopilot State)")
            sections.append("<!-- IMMORTAL: 세션 복원 시 반드시 보존 -->")
            sections.append("")
            sections.append(f"- **활성화**: Yes")
            if ap_state.get("activated_at"):
                sections.append(f"- **활성화 시각**: {ap_state['activated_at']}")
            sections.append(f"- **워크플로우**: {ap_state.get('workflow_name', 'N/A')}")
            sections.append(f"- **현재 단계**: Step {ap_state.get('current_step', '?')}")
            sections.append(f"- **상태**: {ap_state.get('workflow_status', 'N/A')}")
            approved = ap_state.get("auto_approved_steps", [])
            if approved:
                sections.append(f"- **자동 승인된 단계**: {approved}")
            sections.append("")

            # SOT schema validation (P1 — structural integrity)
            schema_warnings = validate_sot_schema(ap_state)
            if schema_warnings:
                sections.append("### SOT 스키마 검증 (Schema Validation)")
                for warning in schema_warnings:
                    sections.append(f"  [WARN] {warning}")
                sections.append("")

            # Per-step output validation (Anti-Skip Guard)
            outputs = ap_state.get("outputs", {})
            if outputs:
                sections.append("### 단계별 산출물 검증 (Anti-Skip Guard)")
                for step_num in sorted(
                    int(k.replace("step-", "")) for k in outputs.keys()
                    if k.startswith("step-")
                ):
                    # FIX-R1+R3: Pass ap_state (flat dict with "outputs" key)
                    # and handle list return type from unified validate_step_output
                    is_valid, l0_warnings = validate_step_output(
                        project_dir, step_num, ap_state
                    )
                    mark = "[OK]" if is_valid else "[FAIL]"
                    reason = l0_warnings[0] if l0_warnings else f"Step {step_num}: OK"
                    sections.append(f"  {mark} {reason}")
                sections.append("")
    except Exception:
        pass  # Non-blocking — autopilot section is supplementary

    # Section 2.55: Quality Gate State (IMMORTAL — conditional, only when gate logs exist)
    # Preserves Verification/pACS/Review state for session recovery during retry/rework
    try:
        gate_lines = _extract_quality_gate_state(project_dir)
        if gate_lines:
            sections.append(SM["quality_gate"])
            sections.append("## 품질 게이트 상태 (Quality Gate State)")
            sections.append(
                "<!-- IMMORTAL: 세션 복원 시 Verification/pACS/Review 재개 맥락 -->"
            )
            sections.append("")
            sections.extend(gate_lines)
            sections.append("")
    except Exception:
        pass  # Non-blocking — quality gate section is supplementary

    # Section 2.6: Active Team State (IMMORTAL — conditional, only when team active)
    try:
        team_state = read_active_team_state(project_dir)
        if team_state:
            sections.append(SM["team"])
            sections.append("## Agent Team 상태 (Active Team State)")
            sections.append("<!-- IMMORTAL: 세션 복원 시 반드시 보존 — RLM Layer 2 -->")
            sections.append("")
            sections.append(f"- **팀 이름**: {team_state['name']}")
            sections.append(f"- **상태**: {team_state['status']}")
            completed = team_state.get("tasks_completed", [])
            pending = team_state.get("tasks_pending", [])
            if completed:
                sections.append(f"- **완료 Task**: {completed}")
            if pending:
                sections.append(f"- **대기 Task**: {pending}")
            sections.append("")

            # Completed summaries (RLM Layer 2 — team work summaries)
            summaries = team_state.get("completed_summaries", {})
            if summaries:
                sections.append("### Teammate 작업 요약 (RLM Layer 2)")
                for task_id, info in summaries.items():
                    if isinstance(info, dict):
                        agent = info.get("agent", "?")
                        model = info.get("model", "?")
                        output = info.get("output", "?")
                        summary = info.get("summary", "")
                        sections.append(f"- **{task_id}** ({agent}, {model}): {output}")
                        if summary:
                            sections.append(f"  - {summary}")
                sections.append("")
    except Exception:
        pass  # Non-blocking — team section is supplementary

    # Section 2.65: ULW State (IMMORTAL — conditional, only when active)
    try:
        ulw_state = detect_ulw_mode(entries)
        if ulw_state:
            sections.append(SM["ulw"])
            sections.append("## ULW 상태 (Ultrawork Mode State)")
            sections.append("<!-- IMMORTAL: 세션 복원 시 반드시 보존 -->")
            sections.append("")
            sections.append(f"- **활성화**: Yes")
            sections.append(f"- **감지 위치**: {ulw_state['detected_in']} user message (index {ulw_state['message_index']})")
            sections.append(f"- **원본 지시**: {_truncate(ulw_state['source_message'], 500)}")

            # Show Autopilot combination state
            ap_state = read_autopilot_state(project_dir)
            if ap_state:
                sections.append(f"- **Autopilot 결합**: Yes (ULW가 Autopilot을 강화 — 재시도 한도 10→15회)")
            else:
                sections.append(f"- **Autopilot 결합**: No (대화형 + ULW)")
            sections.append("")

            sections.append("### ULW 강화 규칙 (Intensifiers)")
            sections.append("1. **I-1. Sisyphus Persistence**: 최대 3회 재시도, 각 시도는 다른 접근법. 100% 완료 또는 불가 사유 보고")
            sections.append("2. **I-2. Mandatory Task Decomposition**: TaskCreate → TaskUpdate → TaskList 필수")
            sections.append("3. **I-3. Bounded Retry Escalation**: 동일 대상 3회 초과 재시도 금지 — 초과 시 사용자 에스컬레이션")
            sections.append("")

            # ULW Compliance Guard — deterministic rule compliance check
            ulw_compliance = check_ulw_compliance(entries)
            if ulw_compliance:
                sections.append("### 준수 상태 (Compliance Guard)")
                sections.append(f"- TaskCreate: {ulw_compliance['task_creates']}회")
                sections.append(f"- TaskUpdate: {ulw_compliance['task_updates']}회")
                sections.append(f"- TaskList: {ulw_compliance['task_lists']}회")
                sections.append(f"- 총 도구 사용: {ulw_compliance['total_tool_uses']}회")
                if ulw_compliance["errors_detected"] > 0:
                    sections.append(f"- 에러 감지: {ulw_compliance['errors_detected']}건")
                    sections.append(f"- 에러 후 조치: {ulw_compliance['post_error_actions']}건")
                if ulw_compliance["max_consecutive_retries"] > 0:
                    sections.append(f"- 최대 연속 재시도: {ulw_compliance['max_consecutive_retries']}회")
                warnings = ulw_compliance.get("warnings", [])
                if warnings:
                    sections.append("")
                    sections.append("**⚠ 강화 규칙 위반 감지:**")
                    for w in warnings:
                        sections.append(f"- {w}")
                else:
                    sections.append("")
                    sections.append("✅ 모든 강화 규칙 준수")
                sections.append("")
    except Exception:
        pass  # Non-blocking — ULW section is supplementary

    # Section 2.6.5: Diagnosis State (IMMORTAL, conditional)
    try:
        diag_dir = os.path.join(project_dir, "diagnosis-logs")
        if os.path.isdir(diag_dir):
            diag_files = sorted([
                f for f in os.listdir(diag_dir) if f.endswith(".md")
            ])
            if diag_files:
                sections.append(SM["diagnosis"])
                sections.append("### Diagnosis State")
                sections.append("<!-- IMMORTAL: 세션 경계에서 진단 맥락 보존 -->")
                sections.append("")
                # Show last 3 diagnosis logs
                for df in diag_files[-3:]:
                    dpath = os.path.join(diag_dir, df)
                    try:
                        with open(dpath, "r", encoding="utf-8") as f:
                            dcontent = f.read(1000)
                        sel = _DIAG_SELECTED_RE.search(dcontent)
                        gate_m = _DIAG_GATE_RE.search(dcontent)
                        sections.append(
                            f"- `{df}`: gate={gate_m.group(1) if gate_m else '?'}, "
                            f"hypothesis={sel.group(1).strip() if sel else '?'}"
                        )
                    except Exception:
                        sections.append(f"- `{df}`: (parse error)")
                sections.append("")
    except Exception:
        pass  # Non-blocking — diagnosis section is supplementary

    # Section 2.7: Design Decisions (C-1 — IMMORTAL, conditional)
    if decisions:
        sections.append(SM["decisions"])
        sections.append(f"{E5_DESIGN_DECISIONS_MARKER} (Design Decisions)")
        sections.append("<!-- IMMORTAL: 세션 복원 시 '왜' 그 결정을 했는지 보존 -->")
        sections.append("")
        for i, dec in enumerate(decisions, 1):
            sections.append(f"{i}. {dec}")
        sections.append("")

    # Section 3: Resume Protocol (deterministic — P1 compliant)
    sections.append(SM["resume"])
    sections.append("## 복원 지시 (Resume Protocol)")
    sections.append("<!-- IMMORTAL: 복원 지시는 세션 복원 시 반드시 보존 — 행동 연속성 핵심 -->")
    sections.append("<!-- Python 결정론적 생성 — P1 준수 -->")
    sections.append("")
    if file_ops:
        sections.append(E5_RICH_CONTENT_MARKER)
        for op in file_ops:
            # C-4: per-file change summary from git diff
            diff_suffix = ""
            if diff_stats:
                # Match by basename or relative path
                rel_path = os.path.relpath(op['path'], project_dir) if os.path.isabs(op['path']) else op['path']
                stats = diff_stats.get(rel_path)
                if not stats:
                    # Try 2-level suffix match (dir/file) to reduce false matches
                    parent = os.path.basename(os.path.dirname(op['path']))
                    basename = os.path.basename(op['path'])
                    suffix_2 = os.path.join(parent, basename) if parent else basename
                    for dp, ds in diff_stats.items():
                        if dp.endswith(suffix_2):
                            stats = ds
                            break
                    # Final fallback: basename-only (accept ambiguity)
                    if not stats:
                        for dp, ds in diff_stats.items():
                            if dp.endswith(basename):
                                stats = ds
                                break
                if stats:
                    diff_suffix = f" (+{stats[0]}/-{stats[1]})"
            sections.append(f"- `{op['path']}` ({op['tool']}, {op['summary']}){diff_suffix}")
    if read_ops:
        sections.append("### 참조하던 파일")
        for op in read_ops[:10]:
            sections.append(f"- `{op['path']}` (Read, {op['count']}회)")
    transcript_size = _get_file_size(entries)
    estimated_tokens = int(transcript_size / CHARS_PER_TOKEN)
    last_tool = ""
    if tool_uses:
        last_tu = tool_uses[-1]
        last_tool_name = last_tu.get("tool_name", "")
        last_tool_path = last_tu.get("file_path", "")
        last_tool = last_tool_name
        if last_tool_path:
            last_tool += f" → {last_tool_path}"
    sections.append("### 세션 정보")
    sections.append(f"- 종료 트리거: {trigger}")
    sections.append(f"- 추정 토큰: ~{estimated_tokens:,}")
    if last_tool:
        sections.append(f"- 마지막 도구: {last_tool}")
    sections.append("")

    # Section 4: Deterministic Completion State (E7 — hallucination prevention)
    sections.append(SM["completion"])
    sections.append(f"{E5_COMPLETION_STATE_MARKER} (Deterministic Completion State)")
    sections.append("<!-- Python 결정론적 생성 — Claude 해석 불필요, 직접 참조 -->")
    sections.append("")
    cs = completion_state
    sections.append("### 도구 호출 결과")
    # Show major tools with success/failure for Edit/Write/Bash
    for tk in ["Edit", "Write", "Bash", "Read", "Task", "Grep", "Glob"]:
        count = cs["tool_counts"].get(tk, 0)
        if count > 0:
            if tk == "Edit":
                sections.append(
                    f"- Edit: {count}회 호출 → {cs['edit_success']} 성공, {cs['edit_fail']} 실패"
                )
            elif tk == "Write":
                sections.append(
                    f"- Write: {count}회 호출 → {cs['write_success']} 성공, {cs['write_fail']} 실패"
                )
            elif tk == "Bash":
                sections.append(
                    f"- Bash: {count}회 호출 → {cs['bash_success']} 성공, {cs['bash_fail']} 실패"
                )
            else:
                sections.append(f"- {tk}: {count}회 호출")
    # Other tools not in the main list
    other_tools = {
        k: v for k, v in cs["tool_counts"].items()
        if k not in ("Edit", "Write", "Bash", "Read", "Task", "Grep", "Glob")
    }
    for name, count in sorted(other_tools.items()):
        sections.append(f"- {name}: {count}회 호출")
    sections.append("")

    if cs["file_verification"]:
        sections.append("### 파일 상태 검증 (저장 시점)")
        sections.append("| 파일 | 존재 | 최종수정 |")
        sections.append("|------|------|---------|")
        for fv in cs["file_verification"]:
            exists_mark = "✓" if fv["exists"] else "✗"
            short_path = os.path.basename(fv["path"])
            sections.append(f"| `{short_path}` | {exists_mark} | {fv['mtime']} |")
        sections.append("")

    if cs["first_timestamp"] or cs["last_timestamp"]:
        sections.append("### 세션 타임라인")
        if cs["first_timestamp"]:
            sections.append(f"- 시작: {cs['first_timestamp']}")
        if cs["last_timestamp"]:
            sections.append(f"- 종료: {cs['last_timestamp']}")
        sections.append("")

    # A6: 최근 도구 호출 시간순 기록 — 에러-복구 패턴 보존
    recent_tools = [
        e for e in entries
        if e.get("type") == "tool_use" and e.get("tool_name")
    ][-10:]  # 마지막 10개
    if recent_tools:
        # Pre-build error lookup: O(n) once instead of O(10n) nested scan
        result_errors = {}
        for e2 in entries:
            if e2.get("type") == "tool_result":
                tid = e2.get("tool_use_id", "")
                if tid and e2.get("is_error"):
                    result_errors[tid] = True
        sections.append("### 최근 도구 활동 (시간순)")
        for rt in recent_tools:
            tool = rt.get("tool_name", "?")
            fp = rt.get("file_path", "")
            ts = rt.get("timestamp", "")[-8:]  # HH:MM:SS
            short_fp = os.path.basename(fp) if fp else ""
            tu_id = rt.get("tool_use_id", "")
            result_tag = " ← ERROR" if result_errors.get(tu_id) else ""
            suffix = f" → `{short_fp}`" if short_fp else ""
            sections.append(f"- [{ts}] {tool}{suffix}{result_tag}")
        sections.append("")

    # Section 5: Git Changes (E2 — ground truth, post-commit aware)
    if any(git_state.values()):
        sections.append(SM["git"])
        sections.append("## Git 변경 상태 (Git Changes)")
        if git_state["status"]:
            sections.append("### Working Tree")
            sections.append(f"```\n{git_state['status']}\n```")
        elif not git_state["diff_stat"]:
            sections.append("### Working Tree")
            sections.append("```\nclean (변경 없음)\n```")
        if git_state["diff_stat"]:
            sections.append("### Uncommitted Changes")
            sections.append(f"```\n{git_state['diff_stat']}\n```")
        if git_state["diff_content"]:
            sections.append("### Diff Detail")
            sections.append(f"```diff\n{git_state['diff_content']}\n```")
        if git_state["recent_commits"]:
            sections.append("### Recent Commits")
            sections.append(f"```\n{git_state['recent_commits']}\n```")
        sections.append("")

    # ━━━ SURVIVAL PRIORITY 2: CRITICAL ━━━

    # Section 6: Modified Files with per-edit details (E3)
    sections.append(SM["modified_files"])
    sections.append("## 수정된 파일 (Modified Files)")
    if file_ops:
        for op in file_ops:
            sections.append(f"### `{op['path']}` ({op['tool']}, {op['summary']})")
            if op.get("details"):
                for j, detail in enumerate(op["details"], 1):
                    sections.append(f"  {j}. {_truncate(detail, 200)}")
            sections.append("")
    else:
        sections.append("(파일 수정 기록 없음)")
    sections.append("")

    # Section 7: Referenced Files
    sections.append(SM["referenced_files"])
    sections.append("## 참조된 파일 (Referenced Files)")
    if read_ops:
        sections.append("| 파일 경로 | 횟수 |")
        sections.append("|----------|------|")
        for op in read_ops[:20]:
            sections.append(f"| `{op['path']}` | {op['count']} |")
    else:
        sections.append("(파일 참조 기록 없음)")
    sections.append("")

    # Section 8: User Messages (verbatim — last N)
    sections.append(SM["user_messages"])
    sections.append("## 사용자 요청 이력 (User Messages)")
    if user_msgs_filtered:
        for i, msg in enumerate(user_msgs_filtered[-12:], 1):
            sections.append(f"{i}. {_truncate(msg['content'], 800)}")
    else:
        sections.append("(사용자 메시지 없음)")
    sections.append("")

    # Section 9: Claude Key Responses (E4 — priority selection, promoted)
    sections.append(SM["responses"])
    sections.append("## Claude 핵심 응답 (Key Responses)")
    meaningful_texts = [
        t for t in assistant_texts
        if len(t["content"]) > 100
    ]
    if meaningful_texts:
        # Priority markers for structured progress reports
        PRIORITY_MARKERS = [
            "Done", "완료", "PASS", "FAIL", "TODO",
            "남은", "진행", "요약", "검증", "수정 완료",
            "## ", "| ", "```",
        ]

        def _priority_score(t):
            content = t["content"]
            score = sum(1 for m in PRIORITY_MARKERS if m in content)
            if len(content) > 500:
                score += 1
            if len(content) > 1000:
                score += 1
            return score

        # Last 3 responses always preserved (most recent context)
        last_3 = meaningful_texts[-3:]
        last_3_ids = set(id(t) for t in last_3)
        # From remaining, select top 5 by priority score
        remaining = [
            t for t in meaningful_texts
            if id(t) not in last_3_ids
        ]
        remaining.sort(key=_priority_score, reverse=True)
        top_priority = remaining[:5]
        # Merge and output in original chronological order
        selected_ids = set(id(t) for t in last_3 + top_priority)
        selected_responses = [t for t in meaningful_texts if id(t) in selected_ids]
        for i, txt in enumerate(selected_responses, 1):
            content = txt["content"]
            if len(content) > 2500:
                # A5: Structure-preserving compression — keep header + conclusion
                # Split: first 1200 chars (intro/structure) + last 1000 chars (conclusion)
                head = content[:1200]
                tail = content[-1000:]
                omitted = len(content) - 2200
                sections.append(f"{i}. {head}\n  [...{omitted}자 생략...]\n  {tail}")
            else:
                sections.append(f"{i}. {content}")
    else:
        sections.append("(Claude 응답 없음)")
    sections.append("")

    # ━━━ SURVIVAL PRIORITY 3: SACRIFICABLE ━━━

    # Section 10: Statistics
    sections.append(SM["statistics"])
    sections.append("## 대화 통계")
    sections.append(f"- 총 메시지: {len(user_msgs_filtered) + len(assistant_texts)}개")
    sections.append(f"- 도구 사용: {len(tool_uses)}회")
    sections.append(f"- 추정 토큰: ~{estimated_tokens:,}")
    sections.append(f"- 저장 트리거: {trigger}")
    if user_msgs_filtered:
        last_msg = _truncate(user_msgs_filtered[-1]["content"], 200)
        sections.append(f"- 마지막 사용자 메시지: \"{last_msg}\"")
    sections.append("")

    # Section 11: Commands Executed
    sections.append(SM["commands"])
    sections.append("## 실행된 명령 (Commands Executed)")
    bash_ops = [t for t in tool_uses if t.get("tool_name") == "Bash"]
    if bash_ops:
        for op in bash_ops[-20:]:
            cmd = _truncate(op.get("command", ""), 150)
            desc = op.get("description", "")
            if cmd:
                sections.append(f"- `{cmd}`" + (f" ({desc})" if desc else ""))
            else:
                sections.append(f"- {op['content']}")
    else:
        sections.append("(명령 실행 기록 없음)")
    sections.append("")

    # Section 12: Work Log Summary
    if work_log:
        sections.append(SM["work_log"])
        sections.append("## 작업 로그 요약 (Work Log Summary)")
        sections.append(f"총 기록: {len(work_log)}개")
        for entry in work_log[-25:]:
            ts = entry.get("timestamp", "")
            tool = entry.get("tool_name", "")
            summary = entry.get("summary", "")
            sections.append(f"- [{ts}] {tool}: {summary}")
        sections.append("")

    # Combine and enforce size limit
    full_md = "\n".join(sections)

    if len(full_md) > MAX_SNAPSHOT_CHARS:
        full_md = _compress_snapshot(full_md, sections)

    return full_md


# =============================================================================
# Deterministic File Operation Extraction
# =============================================================================

def _extract_file_operations(tool_uses, work_log=None):
    """Extract file modification records using structured metadata.

    Uses entry['file_path'] (set by _parse_assistant_entry) instead of
    parsing summary strings. This is 100% deterministic.

    E3 Enhancement: Preserves per-edit details (not just aggregated summary).
    Each edit's OLD→NEW context is stored in 'details' list, preventing
    information loss from aggregation.
    """
    # Track operations per path (preserve insertion order)
    path_order = []
    ops_by_path = {}

    for tu in tool_uses:
        tool_name = tu.get("tool_name", "")

        if tool_name in ("Write", "Edit"):
            # Use structured metadata — NOT string parsing
            path = tu.get("file_path", "")
            if not path:
                continue

            if path not in ops_by_path:
                path_order.append(path)
                ops_by_path[path] = {
                    "count": 0, "last_tool": "", "last_summary": "",
                    "details": [],  # E3: per-edit detail preservation
                }

            record = ops_by_path[path]
            record["count"] += 1
            record["last_tool"] = tool_name

            if tool_name == "Write":
                line_count = tu.get("line_count", 0)
                record["last_summary"] = f"Write ({line_count} lines)"
                record["details"].append(f"Write ({line_count} lines)")
            else:
                record["last_summary"] = "Edit"
                # Extract OLD→NEW detail from content (set by _extract_tool_use_summary)
                content = tu.get("content", "Edit")
                lines = content.split("\n")
                detail_parts = []
                for line in lines[1:3]:  # OLD/NEW lines
                    stripped = line.strip()
                    if stripped:
                        detail_parts.append(stripped)
                detail_str = " | ".join(detail_parts) if detail_parts else "Edit"
                record["details"].append(detail_str)

    # Build result list in insertion order
    ops = []
    for path in path_order:
        record = ops_by_path[path]
        if record["count"] > 1:
            summary = f"{record['last_summary']}, {record['count']}회 수정"
        else:
            summary = record["last_summary"]
        ops.append({
            "path": path,
            "tool": record["last_tool"],
            "summary": summary,
            "details": record["details"],  # E3: per-edit details
        })

    # Supplement from work log (already structured)
    if work_log:
        for entry in work_log:
            path = entry.get("file_path", "")
            if path and path not in ops_by_path:
                ops_by_path[path] = True  # Mark as seen
                ops.append({
                    "path": path,
                    "tool": entry.get("tool_name", ""),
                    "summary": _truncate(entry.get("summary", ""), 100),
                    "details": [],
                })

    return ops


def _extract_read_operations(tool_uses):
    """Extract Read operations with frequency count.

    Deterministic extraction from tool_use entries.
    Tracks which files Claude was consulting during the session.
    Used for Resume Protocol and Knowledge Archive.
    """
    read_counts = {}
    for tu in tool_uses:
        if tu.get("tool_name") == "Read":
            path = tu.get("file_path", "")
            if path:
                read_counts[path] = read_counts.get(path, 0) + 1

    # Sort by frequency (most read first), then alphabetically
    return sorted(
        [{"path": p, "count": c} for p, c in read_counts.items()],
        key=lambda x: (-x["count"], x["path"]),
    )


# =============================================================================
# Token Estimation (Multi-signal)
# =============================================================================

def estimate_tokens(transcript_path, entries=None):
    """
    Multi-signal token estimation.
    Returns (estimated_tokens, signals_dict)
    """
    signals = {}

    # Signal 1: File size
    file_size = 0
    if transcript_path and os.path.exists(transcript_path):
        file_size = os.path.getsize(transcript_path)
    signals["file_size_bytes"] = file_size
    tokens_from_size = int(file_size / CHARS_PER_TOKEN)

    # Signal 2: Message count (if entries available)
    if entries:
        user_count = sum(1 for e in entries if e["type"] == "user_message")
        assistant_count = sum(1 for e in entries if e["type"] == "assistant_text")
        tool_count = sum(1 for e in entries if e["type"] == "tool_use")
        signals["user_messages"] = user_count
        signals["assistant_messages"] = assistant_count
        signals["tool_uses"] = tool_count

        # Heuristic: each substantial exchange ≈ 3-5K tokens
        tokens_from_messages = (user_count + assistant_count) * 2000 + tool_count * 1500
    else:
        tokens_from_messages = tokens_from_size

    # Signal 3: Content character count
    if entries:
        total_chars = sum(len(e.get("content", "")) for e in entries)
        signals["total_content_chars"] = total_chars
        tokens_from_chars = int(total_chars / CHARS_PER_TOKEN)
    else:
        tokens_from_chars = tokens_from_size

    # Weighted average (file size is most reliable)
    estimated = int(
        tokens_from_size * 0.5 +
        tokens_from_messages * 0.25 +
        tokens_from_chars * 0.25
    )

    # Add system overhead
    estimated += SYSTEM_OVERHEAD_TOKENS

    signals["estimated_tokens"] = estimated
    signals["threshold_75"] = THRESHOLD_75_TOKENS
    signals["over_threshold"] = estimated > THRESHOLD_75_TOKENS

    return estimated, signals


# =============================================================================
# File Operations (Atomic + Locking)
# =============================================================================

def atomic_write(filepath, content):
    """Write content atomically: temp file → rename."""
    dirpath = os.path.dirname(filepath)
    os.makedirs(dirpath, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.rename(tmp_path, filepath)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def append_with_lock(filepath, content):
    """Append content with file locking (fcntl.flock)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "a", encoding="utf-8") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(content)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_work_log(snapshot_dir):
    """Load work log entries from JSONL."""
    log_path = os.path.join(snapshot_dir, "work_log.jsonl")
    entries = []
    if not os.path.exists(log_path):
        return entries

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass

    return entries


# =============================================================================
# Dedup Guard
# =============================================================================

def should_skip_save(snapshot_dir, trigger=None):
    """Check if a save was done within dedup window.

    SessionEnd is exempt: /clear is an explicit user action,
    so the save must always happen regardless of dedup window.
    Stop hook uses wider window (30s) to reduce noise.
    """
    if trigger in ("sessionend",):
        return False
    latest_path = os.path.join(snapshot_dir, "latest.md")
    if os.path.exists(latest_path):
        age = time.time() - os.path.getmtime(latest_path)
        # Stop hook uses wider window (30s) to reduce noise
        window = STOP_DEDUP_WINDOW_SECONDS if trigger == "stop" else DEDUP_WINDOW_SECONDS
        if age < window:
            return True
    return False


# =============================================================================
# Snapshot Cleanup
# =============================================================================

def cleanup_snapshots(snapshot_dir):
    """Remove old snapshots, keeping recent ones per trigger type."""
    try:
        files = []
        for f in os.listdir(snapshot_dir):
            if f.endswith(".md") and f != "latest.md":
                fpath = os.path.join(snapshot_dir, f)
                files.append((f, os.path.getmtime(fpath)))

        # Group by trigger type (last part of filename before .md)
        groups = {}
        for fname, mtime in files:
            # Format: YYYYMMDD_HHMMSS_trigger.md
            parts = fname.replace(".md", "").split("_")
            trigger = parts[-1] if len(parts) >= 3 else "unknown"
            if trigger not in groups:
                groups[trigger] = []
            groups[trigger].append((fname, mtime))

        # Keep only MAX per group (sorted by mtime, newest first)
        for trigger, group_files in groups.items():
            max_keep = MAX_SNAPSHOTS.get(trigger, DEFAULT_MAX_SNAPSHOTS)
            group_files.sort(key=lambda x: x[1], reverse=True)
            for fname, _ in group_files[max_keep:]:
                try:
                    os.unlink(os.path.join(snapshot_dir, fname))
                except OSError:
                    pass
    except Exception:
        pass


def _get_file_size(entries):
    """Estimate total character size from entries."""
    total = 0
    for e in entries:
        total += len(e.get("content", ""))
    return total


def _append_compression_audit(content, audit):
    """A5: Append compression audit trail as HTML comment.

    P1 Compliance: Deterministic metadata only.
    Format: single-line HTML comment (invisible in rendered MD, greppable).
    """
    if not audit:
        return content
    final_size = len(content)
    trail = " ".join(audit)
    return content + f"\n<!-- compression-audit: {trail} | final:{final_size}ch/{MAX_SNAPSHOT_CHARS}ch -->"


def _compress_snapshot(full_md, sections):
    """Quality-focused compression (절대 기준 1: 품질 우선).

    Compression priority (sacrifice order — last resort first):
      Phase 1: Deduplicate redundant entries
      Phase 2: Reduce commands section (SACRIFICABLE)
      Phase 3: Reduce work log (SACRIFICABLE)
      Phase 4: Reduce statistics section (SACRIFICABLE)
      Phase 5: Compress Git diff detail (keep stat + commits, drop full diff)
      Phase 6: Compress Claude responses (keep conclusions)
      Phase 7: Hard truncate only as absolute last resort

    Always preserved (IMMORTAL):
      Header, Current Task, Next Step*, SOT, Autopilot State*, ULW State*,
      Team State*, Design Decisions*, Resume Protocol,
      Deterministic Completion State, Git Changes (stat+commits)
      (* = conditional sections, only present when active)

    High priority (CRITICAL):
      Modified Files, Referenced Files, User Messages, Claude Responses
    """
    # A5: Compression audit trail
    audit = []
    original_size = sum(len(s) + 1 for s in sections)  # +1 for \n

    # Phase 1: Deduplicate — remove consecutive identical entries
    deduped_sections = _dedup_sections(sections)
    result = "\n".join(deduped_sections)
    p1_removed = original_size - len(result)
    if p1_removed > 0:
        audit.append(f"P1-dedup:-{p1_removed}ch")
    if len(result) <= MAX_SNAPSHOT_CHARS:
        return _append_compression_audit(result, audit)

    # Phase 2: Compress commands (keep first 3 + last 5)
    prev_size = len(result)
    compressed = _compress_section_entries(
        deduped_sections, "## 실행된 명령", keep_first=3, keep_last=5
    )
    result = "\n".join(compressed)
    p2_removed = prev_size - len(result)
    if p2_removed > 0:
        audit.append(f"P2-cmds:-{p2_removed}ch")
    if len(result) <= MAX_SNAPSHOT_CHARS:
        return _append_compression_audit(result, audit)

    # Phase 3: Compress work log (keep last 10)
    prev_size = len(result)
    compressed = _compress_section_entries(
        compressed, "## 작업 로그 요약", keep_first=0, keep_last=10
    )
    result = "\n".join(compressed)
    p3_removed = prev_size - len(result)
    if p3_removed > 0:
        audit.append(f"P3-wlog:-{p3_removed}ch")
    if len(result) <= MAX_SNAPSHOT_CHARS:
        return _append_compression_audit(result, audit)

    # Phase 4: Remove statistics section entirely (regeneratable)
    prev_size = len(result)
    compressed = _remove_section(compressed, "## 대화 통계")
    result = "\n".join(compressed)
    p4_removed = prev_size - len(result)
    if p4_removed > 0:
        audit.append(f"P4-stats:-{p4_removed}ch")
    if len(result) <= MAX_SNAPSHOT_CHARS:
        return _append_compression_audit(result, audit)

    # Phase 5: Compress Git diff detail (keep stat + commits, drop full diff)
    prev_size = len(result)
    compressed = _remove_section(compressed, "### Diff Detail")
    result = "\n".join(compressed)
    p5_removed = prev_size - len(result)
    if p5_removed > 0:
        audit.append(f"P5-diff:-{p5_removed}ch")
    if len(result) <= MAX_SNAPSHOT_CHARS:
        return _append_compression_audit(result, audit)

    # H10: Extract phase transition markers before response compression
    # Preserve lines containing "→" or "phase" transition indicators in IMMORTAL section
    phase_transition_markers = []
    for line in compressed:
        if ("→" in line and any(kw in line.lower() for kw in
                                ("phase", "research", "implementation", "planning", "orchestration"))) \
                or re.search(r"(?:phase|단계)\s*(?:transition|전환|change)", line, re.I):
            marker_text = line.strip()[:150]
            if marker_text and len(phase_transition_markers) < 3:
                phase_transition_markers.append(
                    f"<!-- PHASE_TRANSITION: {marker_text} -->"
                )

    # Inject phase transition markers into IMMORTAL header area (after first "# Context Recovery" line)
    if phase_transition_markers:
        injected = []
        header_found = False
        for line in compressed:
            injected.append(line)
            if not header_found and line.startswith("# Context Recovery"):
                header_found = True
                for marker in phase_transition_markers:
                    injected.append(marker)
        compressed = injected

    # Phase 6: Compress Claude responses (preserve conclusion — last 300 chars)
    prev_size = len(result)
    compressed = _compress_responses(compressed)
    result = "\n".join(compressed)
    p6_removed = prev_size - len(result)
    if p6_removed > 0:
        audit.append(f"P6-resp:-{p6_removed}ch")
    if len(result) <= MAX_SNAPSHOT_CHARS:
        return _append_compression_audit(result, audit)

    # Phase 7: IMMORTAL-aware hard truncate (absolute last resort)
    # CM-E: Preserve IMMORTAL sections, truncate non-IMMORTAL from bottom up
    # FIX-C1: Marker-first boundary detection — each IMMORTAL marker re-enters
    # IMMORTAL mode even after a non-IMMORTAL section interrupted.
    # Old bug: single flag flipped to False on non-IMMORTAL "## " header,
    # then subsequent IMMORTAL sections were misclassified as non-IMMORTAL.
    immortal_lines = []
    other_lines = []
    in_immortal_section = False
    for line in compressed:
        # IMMORTAL marker always (re-)enters IMMORTAL mode
        # H10: PHASE_TRANSITION markers are also treated as IMMORTAL
        if "<!-- IMMORTAL:" in line or "<!-- PHASE_TRANSITION:" in line:
            in_immortal_section = True
        # Non-IMMORTAL section header exits IMMORTAL mode
        # Must check AFTER marker check so marker on same line wins
        elif line.startswith("## ") and "IMMORTAL" not in line:
            in_immortal_section = False
        if in_immortal_section or line.startswith("# Context Recovery"):
            immortal_lines.append(line)
        else:
            other_lines.append(line)

    immortal_text = "\n".join(immortal_lines)
    other_text = "\n".join(other_lines)
    audit.append(f"P7-truncate:immortal={len(immortal_text)}ch,other={len(other_text)}ch")
    budget = MAX_SNAPSHOT_CHARS - len(immortal_text) - 100
    if budget > 0:
        truncated = immortal_text + "\n" + other_text[:budget] + \
            "\n\n(... 크기 초과로 잘림 — 전체 내역은 sessions/ 아카이브 참조)"
        return _append_compression_audit(truncated, audit)
    # Even IMMORTAL exceeds limit — truncate IMMORTAL itself (preserving start)
    # Reflection fix: Use immortal_text, not Phase 6 result, to avoid
    # cutting mixed content that defeats IMMORTAL-first purpose
    # FIX-C1: Add truncation notice so restored session knows context was cut
    truncation_notice = (
        "\n\n<!-- IMMORTAL: 압축 알림 — 세션 복원 시 핵심 맥락 -->\n"
        "## ⚠ 스냅샷 압축 알림\n"
        "이 스냅샷은 Phase 7 hard truncate를 거쳤습니다. "
        "일부 비-IMMORTAL 섹션(수정 파일 목록, 작업 로그, Claude 응답 등)이 "
        "제거되었을 수 있습니다. 전체 내역은 `sessions/` 아카이브를 참조하세요.\n"
    )
    truncated = immortal_text[:MAX_SNAPSHOT_CHARS - len(truncation_notice) - 100] + \
        truncation_notice
    return _append_compression_audit(truncated, audit)


def _dedup_sections(sections):
    """Remove consecutive duplicate entries within list-style sections."""
    result = []
    prev_line = None
    for line in sections:
        # Skip consecutive identical list items
        if line.startswith("- ") and line == prev_line:
            continue
        result.append(line)
        prev_line = line
    return result


def _compress_section_entries(sections, section_header, keep_first=0, keep_last=5):
    """Compress a specific section's list entries, keeping first N + last N."""
    result = []
    in_section = False
    section_entries = []

    for line in sections:
        if section_header in line:
            in_section = True
            result.append(line)
            continue
        if in_section and line.startswith("##"):
            # End of section — emit compressed entries
            _emit_compressed_entries(result, section_entries, keep_first, keep_last)
            section_entries = []
            in_section = False
            result.append(line)
            continue
        # P1-RLM: SECTION markers adjacent to section headers are part of the section
        if in_section and line.startswith("<!-- SECTION:"):
            _emit_compressed_entries(result, section_entries, keep_first, keep_last)
            section_entries = []
            in_section = False
            result.append(line)
            continue
        if in_section and line.startswith("- "):
            section_entries.append(line)
            continue
        if in_section and not line.strip():
            section_entries.append(line)
            continue
        if in_section:
            # Non-list content in section (e.g., "총 기록: N개")
            result.append(line)
            continue
        result.append(line)

    # If section was the last one
    if section_entries:
        _emit_compressed_entries(result, section_entries, keep_first, keep_last)

    return result


def _emit_compressed_entries(result, entries, keep_first, keep_last):
    """Emit first N + last N entries with omission marker."""
    # Filter out blank lines for counting
    items = [e for e in entries if e.strip()]
    blanks_after = [e for e in entries if not e.strip()]

    total = len(items)
    if total <= keep_first + keep_last:
        result.extend(entries)
        return

    if keep_first > 0:
        result.extend(items[:keep_first])
    omitted = total - keep_first - keep_last
    result.append(f"  (...{omitted}개 항목 생략...)")
    result.extend(items[-keep_last:])
    if blanks_after:
        result.append("")


def _remove_section(sections, section_header):
    """Remove an entire section (header to next ## header) from sections list."""
    result = []
    in_section = False
    # P1-RLM: Track preceding SECTION marker for removal with section
    pending_marker = None
    for line in sections:
        # If this is a SECTION marker, hold it — only emit if not followed by removed section
        if line.startswith("<!-- SECTION:") and not in_section:
            pending_marker = line
            continue
        if section_header in line:
            in_section = True
            pending_marker = None  # Drop the preceding marker too
            continue
        if in_section and line.startswith("## "):
            in_section = False
            result.append(line)
            continue
        if in_section and line.startswith("<!-- SECTION:"):
            # Next section's marker — end current removed section
            in_section = False
            result.append(line)
            continue
        # Emit held marker if it wasn't consumed by a removed section
        if pending_marker is not None:
            result.append(pending_marker)
            pending_marker = None
        # When removing a ### subsection, stop at the next sibling ### header
        if in_section and section_header.startswith("### ") and line.startswith("### ") and section_header not in line:
            in_section = False
            result.append(line)
            continue
        if in_section and line.startswith("### ") and not section_header.startswith("### "):
            # Sub-section within removed ## section — also remove
            continue
        if not in_section:
            result.append(line)
    return result


def _compress_responses(sections):
    """Compress Claude responses: structure-aware compression (C-7).

    Preserves structural markers (headers, lists, code blocks, tables)
    while dropping verbose prose. More generous limits for structured content.
    """
    result = []
    in_section = False

    for line in sections:
        if "## Claude 핵심 응답" in line:
            in_section = True
            result.append(line)
            continue
        if in_section and line.startswith("##"):
            in_section = False
            result.append(line)
            continue
        if in_section and line and line[0].isdigit() and ". " in line[:5]:
            # Numbered response — structure-aware compression
            if len(line) > 500:
                result.append(_structure_aware_compress_line(line))
            else:
                result.append(line)
            continue
        result.append(line)

    return result


def _structure_aware_compress_line(text, max_prefix=120, max_conclusion=400):
    """Compress a single long text line, preserving structural markers (C-7).

    Structure-rich content (headers, lists, tables) gets more generous limits.
    """
    structural_markers = ("## ", "### ", "- ", "* ", "| ", "```", "1. ", "2. ")
    has_structure = any(m in text for m in structural_markers)

    if has_structure:
        # Structured content: keep more context
        prefix = text[:max_prefix]
        conclusion = text[-max_conclusion:]
        return f"{prefix} (...구조 보존...) {conclusion}"
    else:
        # Plain prose: standard compression
        prefix = text[:80]
        conclusion = text[-300:]
        return f"{prefix} (...) {conclusion}"


def get_snapshot_dir(project_dir=None):
    """Get the context-snapshots directory path."""
    if not project_dir:
        project_dir = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
    return os.path.join(project_dir, ".claude", "context-snapshots")


def read_stdin_json():
    """Read and parse JSON from stdin (hook input)."""
    try:
        raw = sys.stdin.read()
        if raw.strip():
            return json.loads(raw)
    except (json.JSONDecodeError, Exception):
        pass
    return {}


# =============================================================================
# E5 Guard Helper (A1: Multi-Signal Rich Content Detection)
# =============================================================================

def is_rich_snapshot(content):
    """Multi-signal rich content detection for E5 Empty Snapshot Guard.

    P1 Compliance: Deterministic — size threshold + marker counting.
    Returns True if snapshot is "rich" (should not be overwritten by empty one).

    Signals (any 2 of the following):
      1. Content length >= 3KB (aligned with E6 MIN_QUALITY_SIZE)
      2-4. Presence of E5_RICH_SIGNALS markers
    """
    if not content:
        return False

    signal_count = 0

    # Signal 1: Size threshold (aligned with E6 MIN_QUALITY_SIZE = 3000)
    if len(content.encode("utf-8")) >= 3000:
        signal_count += 1

    # Signals 2-4: Section markers
    for marker in E5_RICH_SIGNALS:
        if marker in content:
            signal_count += 1

    return signal_count >= 2


# =============================================================================
# E5 Guard + Knowledge Archive — Consolidated Helpers
# =============================================================================

def update_latest_with_guard(snapshot_dir, md_content, entries):
    """Atomically update latest.md with E5 Empty Snapshot Guard.

    Returns True if latest.md was updated, False if existing rich snapshot
    was protected from overwrite by an empty one.

    P1 Compliance: Deterministic (tool_use count + is_rich_snapshot).
    SOT Compliance: No SOT access.
    """
    latest_path = os.path.join(snapshot_dir, "latest.md")
    new_tool_count = sum(1 for e in entries if e.get("type") == "tool_use")

    if os.path.exists(latest_path) and new_tool_count == 0:
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                existing_content = f.read()
            if is_rich_snapshot(existing_content):
                return False
        except Exception:
            pass

    atomic_write(latest_path, md_content)
    return True


def archive_and_index_session(
    snapshot_dir, md_content, session_id, trigger,
    project_dir, entries, transcript_path,
):
    """Archive snapshot + extract knowledge-index facts + cleanup.

    Consolidates the 3-step archive pattern used by all save triggers:
      1. Archive snapshot to sessions/ directory
      2. Extract session facts → knowledge-index.jsonl
      3. Rotate archives and index

    P1 Compliance: All operations deterministic.
    SOT Compliance: Read-only SOT access (via extract_session_facts).
    Timestamp format: ISO-like %Y-%m-%dT%H%M%S (unified across all triggers).
    """
    # Step 1: Archive to sessions/ (isolated — failure does NOT block Step 2)
    # RLM rationale: archive is backup; knowledge-index is the RLM-critical asset.
    # If sessions/ mkdir or write fails, Step 2 must still record the session.
    try:
        sessions_dir = os.path.join(snapshot_dir, "sessions")
        os.makedirs(sessions_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        archive_name = f"{ts}_{session_id[:8]}.md"
        archive_path = os.path.join(sessions_dir, archive_name)
        atomic_write(archive_path, md_content)
    except Exception:
        pass  # Non-blocking — Step 2 (RLM-critical) proceeds independently

    # Step 2: Extract session facts → knowledge-index.jsonl (RLM-critical)
    try:
        estimated_tokens, _ = estimate_tokens(transcript_path, entries)
        facts = extract_session_facts(
            session_id=session_id,
            trigger=trigger,
            project_dir=project_dir,
            entries=entries,
            token_estimate=estimated_tokens,
        )
        ki_path = os.path.join(snapshot_dir, "knowledge-index.jsonl")
        replace_or_append_session_facts(ki_path, facts)
    except Exception:
        pass  # Non-blocking

    # Step 3: Rotate archives and index (each cleanup is internally protected)
    cleanup_session_archives(snapshot_dir)
    cleanup_knowledge_index(snapshot_dir)


# =============================================================================
# Path Tag Extraction (A3: Language-Independent Search Tags)
# =============================================================================

def extract_path_tags(file_paths):
    """Extract language-independent search tags from file paths.

    P1 Compliance: Deterministic string processing only.
    Returns: sorted unique list of tag strings (max 20).

    Tag sources:
      - CamelCase splitting: "AuthService.py" → ["auth", "service"]
      - snake_case splitting: "user_auth.py" → ["user", "auth"]
      - Extension mapping: ".py" → "python"
    """
    tags = set()
    for fp in file_paths:
        if not fp:
            continue
        parts = Path(fp).parts
        for part in parts:
            name = Path(part).stem  # filename without extension
            if name.startswith(".") or name in _PATH_SKIP_NAMES:
                continue
            # CamelCase splitting: "AuthService" → ["Auth", "Service"]
            # Also handles: "getHTTPResponse" → ["get", "HTTP", "Response"]
            subtokens = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', name)
            for st in subtokens:
                lower = st.lower()
                if len(lower) >= 3:  # skip noise ("a", "db", "io")
                    tags.add(lower)
        # Extension tag
        ext = os.path.splitext(fp)[1].lower()
        if ext in _EXT_TAGS:
            tags.add(_EXT_TAGS[ext])
    return sorted(tags)[:20]


# =============================================================================
# Knowledge-Index Schema Validation (P1: Hallucination Prevention)
# =============================================================================

# RLM-critical keys that MUST exist in every knowledge-index entry.
# If extract_session_facts() is modified and accidentally drops a key,
# this validation fills safe defaults — writing incomplete data is better
# than writing nothing (RLM visibility > field completeness).
_KI_REQUIRED_DEFAULTS = {
    "session_id": "",
    "timestamp": "",
    "user_task": "",
    "modified_files": [],
    "read_files": [],
    "tools_used": {},
    "final_status": "unknown",
    "tags": [],
    "phase": "",
    "completion_summary": {},
    "diagnosis_patterns": [],
    "thesis_step": None,  # CM-3: thesis current_step at archive time (int or None)
    "gate_results": {},  # R-12: cross-session gate pass/fail tracking (e.g. {"gate-1": "pass"})
    "invocation_number": None,  # B-3: active invocation number at archive time (int or None)
    # QO-5: Quality context fields for cross-session quality optimization
    "previous_section_outputs": [],  # QO-5a: titles + word counts of last N outputs
    "review_feedback_summary": "",  # QO-5b: latest review feedback summary
    "word_count_trend": [],  # QO-5c: per-step word counts for pacing
}


def _validate_session_facts(facts):
    """P1 Hallucination Prevention: Ensure RLM-critical keys exist before write.

    Deterministic schema enforcement — fills missing keys with safe defaults.
    Prevents malformed knowledge-index entries from breaking RLM queries like:
      Grep "tags.*python" knowledge-index.jsonl
      Grep "final_status.*success" knowledge-index.jsonl

    Returns: facts dict with all required keys guaranteed present.
    """
    for key, default_val in _KI_REQUIRED_DEFAULTS.items():
        if key not in facts:
            # Create new mutable instances to avoid shared references
            if isinstance(default_val, list):
                facts[key] = []
            elif isinstance(default_val, dict):
                facts[key] = {}
            else:
                facts[key] = default_val
    return facts


# =============================================================================
# Knowledge Archive (Area 1: Cross-Session Knowledge Archive)
# =============================================================================

def _classify_error_patterns(entries):
    """CM-1: Classify error patterns from tool results for cross-session learning.

    P1 Compliance: Regex-based deterministic classification.
    A2 Enhancement: File-aware, window-limited resolution matching.
    Returns: list of {"type": str, "tool": str, "file": str, "resolution": dict|None} (max 5).
    """
    tool_results = [e for e in entries if e["type"] == "tool_result"]
    tool_uses = [e for e in entries if e["type"] == "tool_use"]

    # Build tool_use_id → tool_name mapping
    id_to_tool = {tu.get("tool_use_id", ""): tu.get("tool_name", "") for tu in tool_uses}
    id_to_file = {tu.get("tool_use_id", ""): tu.get("file_path", "") for tu in tool_uses}

    # CM-B + E-1: Expanded error taxonomy — reduces "unknown" classification from ~80% to ~30%
    # D-7: Type names MUST match _RISK_WEIGHTS keys (~line 127). Adding a new type here
    #       without a corresponding _RISK_WEIGHTS entry → fallback weight 0.7 applied.
    ERROR_TAXONOMY = [
        ("file_not_found", re.compile(r"No such file|FileNotFoundError|ENOENT|not found", re.I)),
        ("permission", re.compile(r"Permission denied|EACCES|PermissionError|EPERM", re.I)),
        ("syntax", re.compile(r"SyntaxError|syntax error|parse error|unexpected token", re.I)),
        ("timeout", re.compile(r"timed? ?out|TimeoutError|deadline exceeded|ETIMEDOUT", re.I)),
        ("dependency", re.compile(r"ModuleNotFoundError|ImportError|Cannot find module|require\(\) failed", re.I)),
        # B-4: Added re.DOTALL — "old_string ... not found" may span multiple lines
        ("edit_mismatch", re.compile(r"old_string.*not found|not unique|no match|string not found in file", re.I | re.DOTALL)),
        # E-1: New patterns (Reflection: tightened to reduce false positives)
        ("type_error", re.compile(r"TypeError|type error|undefined is not a function|\w+ is not a function(?! of\b)", re.I)),
        ("value_error", re.compile(r"ValueError|invalid (?:value|argument|literal)|value.{0,30}out of range", re.I)),
        ("connection", re.compile(r"ConnectionError|ECONNREFUSED|ECONNRESET|network error|fetch failed", re.I)),
        ("memory", re.compile(r"MemoryError|out of memory|heap (?:space|memory|allocation|overflow)|ENOMEM|allocation failed", re.I)),
        ("git_error", re.compile(r"fatal:.*git|merge conflict|CONFLICT|not a git repository", re.I | re.DOTALL)),
        ("command_not_found", re.compile(r"command not found|not recognized|is not recognized", re.I)),
    ]

    # A2: Build position map for resolution matching (file-aware, window-limited)
    entry_id_to_pos = {}
    for i, e in enumerate(entries):
        entry_id_to_pos[id(e)] = i

    patterns = []
    for tr in tool_results:
        if not tr.get("is_error", False):
            continue
        content = tr.get("content", "")[:500]
        tid = tr.get("tool_use_id", "")
        error_type = "unknown"
        for etype, regex in ERROR_TAXONOMY:
            if regex.search(content):
                error_type = etype
                break

        # A2: Resolution matching — find successful follow-up within 10 entries
        # Extended from 5→10 to capture multi-retry recovery chains
        resolution = None
        retry_count = 0
        error_file = os.path.basename(id_to_file.get(tid, ""))
        err_pos = entry_id_to_pos.get(id(tr), -1)
        if err_pos >= 0:
            for next_e in entries[err_pos + 1 : err_pos + 11]:
                if next_e.get("type") != "tool_result":
                    continue
                next_tid = next_e.get("tool_use_id", "")
                next_tool = id_to_tool.get(next_tid, "")
                next_file = os.path.basename(id_to_file.get(next_tid, ""))
                if next_e.get("is_error", False):
                    # Count intermediate failures (retry attempts)
                    if next_tool in ("Edit", "Write", "Bash"):
                        retry_count += 1
                    continue
                # File-aware: same file must match (or error had no file context)
                if next_tool in ("Edit", "Write", "Bash") and (
                    not error_file or next_file == error_file
                ):
                    resolution = {"tool": next_tool, "file": next_file}
                    if retry_count > 0:
                        resolution["retries"] = retry_count
                    break

        patterns.append({
            "type": error_type,
            "tool": id_to_tool.get(tid, ""),
            "file": error_file,
            "resolution": resolution,
        })

    return patterns[:5]


def _extract_success_patterns(entries):
    """Extract successful tool sequence patterns for cross-session learning.

    Detects "Edit/Write → successful Bash" sequences — the canonical pattern
    for code modification followed by validation (e.g., tests, builds).

    P1 Compliance: Deterministic extraction from transcript entries.
    Returns: list of {"sequence": str, "files": list, "bash_cmd": str} (max 5).
    """
    tool_uses = [e for e in entries if e["type"] == "tool_use"]
    tool_results = [e for e in entries if e["type"] == "tool_result"]

    # Build result lookup
    result_by_id = {}
    for tr in tool_results:
        tid = tr.get("tool_use_id", "")
        if tid:
            result_by_id[tid] = tr.get("is_error", False)

    patterns = []
    # Sliding window: track consecutive Edit/Write, then look for successful Bash
    edit_buffer = []  # (tool_name, file_path)

    for tu in tool_uses:
        name = tu.get("tool_name", "")
        tid = tu.get("tool_use_id", "")
        is_err = result_by_id.get(tid, False)

        if name in ("Edit", "Write") and not is_err:
            fp = tu.get("file_path", "")
            edit_buffer.append((name, os.path.basename(fp) if fp else ""))
        elif name == "Bash" and not is_err and edit_buffer:
            # Successful Bash after Edit/Write sequence — capture pattern
            cmd = tu.get("command", "")[:100]
            seq_parts = [f"{t[0]}" for t in edit_buffer[-5:]]  # Last 5 edits
            seq_parts.append("Bash")
            files = sorted(set(t[1] for t in edit_buffer[-5:] if t[1]))
            patterns.append({
                "sequence": "→".join(seq_parts),
                "files": files[:5],
                "bash_cmd": cmd,
            })
            edit_buffer = []  # Reset buffer after capture
        elif name not in ("Edit", "Write", "Read"):
            # Non-Edit/Write/Read tool breaks the sequence (Read is transparent)
            if name != "Bash":
                edit_buffer = []

    return patterns[:5]


def _extract_hypothesis_graveyard(work_log_entries):
    """H3: Extract tried-and-failed approaches from work log.

    Looks for patterns in assistant responses that indicate:
    - "tried X but failed" / "doesn't work" / "wrong approach"
    - "considered X" / "alternatively" followed by rejection

    P1 Compliance: Deterministic regex-based extraction.
    Returns: list of max 5 entries:
        [{"text": "...", "status": "tried|considered", "outcome": "..."}]
    """
    # Patterns indicating tried-and-failed approaches
    _TRIED_PATTERNS = [
        # "tried X but failed/didn't work"
        re.compile(
            r"(?:tried|attempted|tested)\s+(.{10,80}?)\s+(?:but|however)\s+(.{5,80}?)(?:\.|$)",
            re.I,
        ),
        # "X doesn't work / didn't work / won't work"
        re.compile(
            r"(.{10,80}?)\s+(?:doesn't|didn't|does not|did not|won't|will not)\s+work(.{0,60}?)(?:\.|$)",
            re.I,
        ),
        # "wrong approach / incorrect approach"
        re.compile(
            r"(?:wrong|incorrect|bad|failed)\s+(?:approach|method|strategy)\s*[:—-]?\s*(.{10,80}?)(?:\.|$)",
            re.I,
        ),
    ]
    _CONSIDERED_PATTERNS = [
        # "considered X but" / "alternatively X ... however"
        re.compile(
            r"(?:considered|evaluated|explored)\s+(.{10,80}?)\s+(?:but|however)\s+(.{5,80}?)(?:\.|$)",
            re.I,
        ),
        # "ruled out X" / "rejected X" / "discarded X"
        re.compile(
            r"(?:ruled out|rejected|discarded|abandoned)\s+(.{10,80}?)(?:\s+(?:because|due to|since)\s+(.{5,80}?))?(?:\.|$)",
            re.I,
        ),
    ]

    results = []
    assistant_texts = [
        e for e in work_log_entries
        if e.get("type") == "assistant_text"
    ]

    for entry in assistant_texts:
        content = entry.get("content", "")
        if not content:
            continue

        # Check tried patterns
        for pat in _TRIED_PATTERNS:
            for m in pat.finditer(content):
                text = m.group(1).strip()
                outcome = m.group(2).strip() if m.lastindex >= 2 else "failed"
                results.append({
                    "text": text[:120],
                    "status": "tried",
                    "outcome": outcome[:120],
                })
                if len(results) >= 5:
                    return results

        # Check considered patterns
        for pat in _CONSIDERED_PATTERNS:
            for m in pat.finditer(content):
                text = m.group(1).strip()
                outcome = m.group(2).strip() if m.lastindex >= 2 and m.group(2) else "rejected"
                results.append({
                    "text": text[:120],
                    "status": "considered",
                    "outcome": outcome[:120],
                })
                if len(results) >= 5:
                    return results

    return results[:5]


def _extract_pacs_from_sot(project_dir):
    """CM-1: Extract pACS min-score from SOT (read-only).

    P1 Compliance: Deterministic YAML/regex extraction.
    SOT Compliance: Read-only access.
    Returns: int or None.
    """
    if not project_dir:
        return None
    try:
        import yaml
        for sp in sot_paths(project_dir):
            if os.path.exists(sp) and not sp.endswith(".json"):
                with open(sp, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f.read())
                if isinstance(data, dict):
                    wf = data.get("workflow", {})
                    if isinstance(wf, dict):
                        pacs = wf.get("pacs", {})
                        if isinstance(pacs, dict) and "min_score" in pacs:
                            return pacs["min_score"]
    except Exception:
        pass
    return None


def _extract_team_summaries(project_dir):
    """FIX-H1: Extract active_team.completed_summaries from SOT (read-only).

    Preserves team coordination history in knowledge-index.jsonl,
    surviving snapshot rotation and Phase 6-7 compression.

    P1 Compliance: Deterministic YAML extraction.
    SOT Compliance: Read-only access.
    FIX-R4: Removed .json filter — yaml.safe_load() can parse JSON (JSON ⊂ YAML).
    Returns: dict or None.
    """
    if not project_dir:
        return None
    try:
        import yaml
        for sp in sot_paths(project_dir):
            if os.path.exists(sp):
                with open(sp, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f.read())
                if isinstance(data, dict):
                    wf = data.get("workflow", {})
                    if isinstance(wf, dict):
                        active_team = wf.get("active_team", {})
                        if isinstance(active_team, dict):
                            summaries = active_team.get("completed_summaries", {})
                            if summaries:
                                return summaries
    except Exception:
        pass
    return None


def _get_step_gate_deps():
    """Get step→gate dependency map from checklist_manager (SOT) with fallback.

    Tries dynamic import of checklist_manager.STEP_DEPENDENCIES first (SOT).
    Falls back to inline constant if import fails (e.g., circular import).

    P1 Compliance: Deterministic — returns same dict regardless of path.
    """
    try:
        # Dynamic import — avoids top-level circular dependency
        import importlib
        cm = importlib.import_module("checklist_manager")
        step_deps = getattr(cm, "STEP_DEPENDENCIES", None)
        if isinstance(step_deps, dict):
            # Extract only gate requirements (not hitl/phase)
            return {
                phase: dep["gate"]
                for phase, dep in step_deps.items()
                if isinstance(dep, dict) and "gate" in dep
            }
    except Exception:
        pass
    # Fallback: inline subset (last resort — synced with checklist_manager.py)
    return {
        "wave-2": "gate-1", "wave-3": "gate-2",
        "wave-4": "gate-3", "wave-5": "srcs-full",
    }


def _extract_thesis_continuity(project_dir):
    """Phase 1-A: Extract pending gates and blocked steps from thesis SOT.

    Iterates thesis-output/{project_name}/session.json (multiple projects).
    Aggregates across all active thesis projects.

    P1 Compliance: Deterministic JSON extraction.
    SOT Compliance: Read-only access.
    Non-blocking: returns None on any error.
    """
    if not project_dir:
        return None
    try:
        thesis_root = os.path.join(project_dir, "thesis-output")
        if not os.path.isdir(thesis_root):
            return None

        all_pending_gates = []
        all_blocked_steps = []
        step_deps = _get_step_gate_deps()

        for proj_name in sorted(os.listdir(thesis_root)):
            sot_path = os.path.join(thesis_root, proj_name, "session.json")
            if not os.path.isfile(sot_path):
                continue
            try:
                with open(sot_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict):
                continue

            # Skip completed/paused projects
            status = data.get("status", "")
            if status in ("completed", "paused"):
                continue

            gates = data.get("gates", {})
            if not isinstance(gates, dict):
                continue

            # Pending gates: gates with status != "pass"
            for gname, gdata in gates.items():
                if isinstance(gdata, dict) and gdata.get("status") != "pass":
                    label = f"{gname} ({proj_name})" if proj_name else gname
                    all_pending_gates.append(label)

            # Blocked steps: phases whose required gate hasn't passed
            for phase, required_gate in step_deps.items():
                gate_data = gates.get(required_gate, {})
                if isinstance(gate_data, dict) and gate_data.get("status") != "pass":
                    all_blocked_steps.append(
                        f"{phase} (requires {required_gate}, {proj_name})"
                    )

        result = {}
        if all_pending_gates:
            result["pending_gates"] = sorted(all_pending_gates)
        if all_blocked_steps:
            result["blocked_steps"] = sorted(all_blocked_steps)
        return result if result else None
    except Exception:
        return None


def _classify_session_type(user_task, last_instruction, phase):
    """Phase 1-B: Classify session type deterministically.

    Categories: debugging, feature, refactoring, audit, research, writing, translation.
    Uses keyword matching on user_task + last_instruction.

    P1 Compliance: Pure regex/string matching — no LLM inference.
    Word boundaries (\b) prevent false positives (e.g., "prefix" ≠ "fix").
    Korean keywords don't need \b (Korean characters are inherently boundary-forming).
    Returns: string category or empty string.
    """
    text = f"{user_task} {last_instruction}".lower()

    # Priority-ordered pattern matching
    # English patterns use \b word boundaries; Korean patterns match as-is
    patterns = [
        ("debugging", [r"\bbug\b", r"\bfix\b", r"\berror\b", r"에러", r"디버그", r"오류"]),
        ("audit", [r"\baudit\b", r"검수", r"성찰", r"전수조사", r"\binspect", r"\breview\b"]),
        ("refactoring", [r"\brefactor", r"리팩", r"\bclean.?up\b", r"정리", r"\breorganiz"]),
        ("translation", [r"\btranslat", r"번역", r"\bglossary\b", r"용어"]),
        ("writing", [r"\bwrit(?:e|ing)\b", r"작성", r"\bdraft\b", r"논문", r"\bthesis\b", r"\bchapter\b"]),
        ("research", [r"\bresearch", r"연구", r"\bliterature\b", r"문헌", r"\bsurvey\b"]),
        ("feature", [r"\bfeat", r"\badd\b", r"\bimplement", r"구현", r"추가", r"생성", r"\bcreate\b"]),
    ]
    for category, keywords in patterns:
        for kw in keywords:
            if re.search(kw, text):
                return category

    # Fallback: infer from phase if no keyword match
    phase_map = {
        "exploration": "research",
        "implementation": "feature",
        "debugging": "debugging",
    }
    return phase_map.get(phase, "")


def _extract_gate_results_snapshot(project_dir):
    """R-12: Extract gate pass/fail snapshot for cross-session trend detection.

    Reads thesis SOT gates block and returns compact pass/fail dict.
    Includes completed/paused projects — their gate outcomes are valuable
    for cross-session gate trend analysis and quality pattern detection.

    P1 Compliance: Deterministic JSON read, read-only.
    Returns: dict like {"gate-1": "pass", "gate-2": "fail"} or None.
    """
    try:
        thesis_root = os.path.join(project_dir, "thesis-output")
        if not os.path.isdir(thesis_root):
            return None
        for proj_name in sorted(os.listdir(thesis_root)):
            sot_path = os.path.join(thesis_root, proj_name, "session.json")
            if not os.path.isfile(sot_path):
                continue
            try:
                with open(sot_path, "r", encoding="utf-8") as f:
                    sot = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            gates = sot.get("gates", {})
            if not isinstance(gates, dict) or not gates:
                continue
            result = {}
            for gname, gdata in gates.items():
                if isinstance(gdata, dict):
                    result[gname] = gdata.get("status", "pending")
                else:
                    result[gname] = str(gdata)
            if result:
                return result
        return None
    except Exception:
        return None


def _extract_thesis_step_at_archive(project_dir):
    """CM-3: Read current_step from thesis session.json at archive time.

    Iterates thesis-output/{project_name}/session.json. Returns the step
    number of the most recent thesis project (by highest current_step).
    Includes completed/paused projects — their final step is valuable for
    RLM proximity scoring and cross-session context.

    P1 Compliance: Deterministic JSON extraction.
    SOT Compliance: Read-only access.
    Non-blocking: returns None on any error or if no thesis project.
    Returns: int (current_step) or None
    """
    if not project_dir:
        return None
    try:
        thesis_root = os.path.join(project_dir, "thesis-output")
        if not os.path.isdir(thesis_root):
            return None
        best_step = None
        for proj_name in sorted(os.listdir(thesis_root)):
            sot_path = os.path.join(thesis_root, proj_name, "session.json")
            if not os.path.isfile(sot_path):
                continue
            try:
                with open(sot_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict):
                continue
            step = data.get("current_step")
            if isinstance(step, int) and (best_step is None or step > best_step):
                best_step = step
        return best_step
    except Exception:
        return None


def extract_session_facts(session_id, trigger, project_dir, entries, token_estimate=0):
    """Extract deterministic session facts for knowledge-index.jsonl.

    P1 Compliance: All fields are deterministic extractions.
    No semantic inference, no heuristic judgment.
    """
    user_messages = [e for e in entries if e["type"] == "user_message"]
    tool_uses = [e for e in entries if e["type"] == "tool_use"]

    # First user message (C-2: expanded to 300 chars for richer cross-session context)
    user_task = ""
    if user_messages:
        # Skip system-injected messages
        for msg in user_messages:
            content = msg.get("content", "")
            if not (content.startswith("<") and ">" in content[:50]):
                user_task = content[:300]
                break

    # Last user instruction (deterministic) — 품질 최적화
    # 긴 세션에서 마지막 지시가 "현재 작업 상태"를 더 정확히 반영한다.
    last_instruction = ""
    if user_messages:
        for msg in reversed(user_messages):
            content = msg.get("content", "")
            if not (content.startswith("<") and ">" in content[:50]):
                if content[:300] != user_task:  # 첫 메시지와 동일하면 생략
                    last_instruction = content[:300]
                break

    # Modified files — unique paths from Write/Edit
    modified_files = sorted(set(
        tu.get("file_path", "") for tu in tool_uses
        if tu.get("tool_name") in ("Write", "Edit") and tu.get("file_path")
    ))

    # B2: Per-file modification metadata — tool type + edit count for change magnitude
    file_detail = {}
    for tu in tool_uses:
        tool_name = tu.get("tool_name", "")
        fp = tu.get("file_path", "")
        if tool_name in ("Write", "Edit") and fp:
            if fp not in file_detail:
                file_detail[fp] = {"tool": tool_name, "edits": 0}
            file_detail[fp]["edits"] += 1
            # Write overwrites; if both Write and Edit occurred, record Write
            if tool_name == "Write":
                file_detail[fp]["tool"] = "Write"

    # Read files — unique paths from Read
    read_files = sorted(set(
        tu.get("file_path", "") for tu in tool_uses
        if tu.get("tool_name") == "Read" and tu.get("file_path")
    ))

    # Tool usage counts (deterministic)
    tools_used = {}
    for tu in tool_uses:
        name = tu.get("tool_name", "unknown")
        tools_used[name] = tools_used.get(name, 0) + 1

    # CM-D + E-3: Tool sequence — consecutive distinct tool names (run-length compressed)
    # Captures work patterns like "Read→Read→Edit→Bash→Read→Edit" → "Read(2)→Edit→Bash→Read→Edit"
    tool_sequence_parts = []
    tool_sequence_with_files_parts = []  # H5: includes file basenames
    prev_tool = None
    count = 0
    segment_files = []  # H5: track file paths per RLE segment
    for tu in tool_uses:
        name = tu.get("tool_name", "unknown")
        if name == prev_tool:
            count += 1
            fp = tu.get("file_path", "")
            if fp:
                bn = os.path.basename(fp)
                if bn and bn not in segment_files:
                    segment_files.append(bn)
        else:
            if prev_tool:
                tool_sequence_parts.append(f"{prev_tool}({count})" if count > 1 else prev_tool)
                # H5: Build file-annotated segment
                file_hint = ",".join(segment_files[:2])
                if file_hint:
                    if count > 1:
                        tool_sequence_with_files_parts.append(
                            f"{prev_tool}({count},[{file_hint}])")
                    else:
                        tool_sequence_with_files_parts.append(
                            f"{prev_tool}([{file_hint}])")
                else:
                    tool_sequence_with_files_parts.append(
                        f"{prev_tool}({count})" if count > 1 else prev_tool)
            prev_tool = name
            count = 1
            segment_files = []
            fp = tu.get("file_path", "")
            if fp:
                bn = os.path.basename(fp)
                if bn:
                    segment_files.append(bn)
    if prev_tool:
        tool_sequence_parts.append(f"{prev_tool}({count})" if count > 1 else prev_tool)
        file_hint = ",".join(segment_files[:2])
        if file_hint:
            if count > 1:
                tool_sequence_with_files_parts.append(
                    f"{prev_tool}({count},[{file_hint}])")
            else:
                tool_sequence_with_files_parts.append(
                    f"{prev_tool}([{file_hint}])")
        else:
            tool_sequence_with_files_parts.append(
                f"{prev_tool}({count})" if count > 1 else prev_tool)
    tool_sequence = "→".join(tool_sequence_parts[-30:])  # Last 30 segments to cap size

    # H5: tool_sequence_with_files — file-annotated RLE, capped at 500 chars
    tsf_parts = tool_sequence_with_files_parts[-30:]
    tool_sequence_with_files = "→".join(tsf_parts)
    while len(tool_sequence_with_files) > 500 and tsf_parts:
        tsf_parts.pop(0)  # Drop oldest segments to fit cap
        tool_sequence_with_files = "→".join(tsf_parts)

    # B-3: Phase detection — current dominant phase
    phase = detect_conversation_phase(tool_uses)

    # B-3: Primary language detection (deterministic — file extension counting)
    ext_counts = {}
    all_files = modified_files + read_files
    for fp in all_files:
        ext = os.path.splitext(fp)[1].lower()
        if ext:
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
    primary_language = ""
    if ext_counts:
        primary_language = max(ext_counts, key=ext_counts.get)

    # B-3: Phase transitions (multi-phase detection, with tool_count per phase)
    transitions = detect_phase_transitions(tool_uses)
    if len(transitions) > 1:
        phase_flow = " → ".join(
            f"{t[0]}({t[2]-t[1]})" for t in transitions
        )
    else:
        phase_flow = phase

    facts = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "project": project_dir,
        "user_task": user_task,
        "modified_files": modified_files,
        "modified_files_detail": file_detail,  # B2: per-file tool + edit count
        "read_files": read_files,
        "tools_used": tools_used,
        "trigger": trigger,
        "token_estimate": token_estimate,
        "phase": phase,
        "phase_flow": phase_flow,
        "primary_language": primary_language,
        "tool_sequence": tool_sequence,  # CM-D + E-3: work pattern analysis
        "tool_sequence_with_files": tool_sequence_with_files,  # H5: file-annotated RLE
    }

    # A4: Search tags — language-independent path-derived keywords for RLM probing
    all_paths = modified_files + read_files
    search_tags = extract_path_tags(all_paths)
    if search_tags:
        facts["tags"] = search_tags

    if last_instruction:
        facts["last_instruction"] = last_instruction

    # E7 + E2: Completion state and git summary (deterministic, reuses existing functions)
    completion = extract_completion_state(entries, project_dir)
    git_state = capture_git_state(project_dir, max_diff_chars=500)

    facts["completion_summary"] = {
        "total_tool_calls": completion["total_tool_calls"],
        "edit_success": completion["edit_success"],
        "edit_fail": completion["edit_fail"],
        "bash_success": completion["bash_success"],
        "bash_fail": completion["bash_fail"],
    }
    facts["git_summary"] = git_state.get("status", "")[:200]

    # E-4: final_status — deterministic session outcome classification
    total_fails = completion["edit_fail"] + completion["bash_fail"]
    total_success = completion["edit_success"] + completion["bash_success"]
    if total_fails == 0 and total_success > 0:
        facts["final_status"] = "success"
    elif total_fails > 0 and total_success > total_fails:
        facts["final_status"] = "incomplete"  # Some failures but mostly succeeded
    elif total_fails > 0:
        facts["final_status"] = "error"
    else:
        facts["final_status"] = "unknown"  # No edits/bash at all (read-only session)

    # Session duration (deterministic timestamp difference)
    timestamps = [e.get("timestamp", "") for e in entries if e.get("timestamp")]
    if len(timestamps) >= 2:
        facts["session_duration_entries"] = len(timestamps)

    # CM-1: Cross-session knowledge enrichment fields
    # 1. Design decisions — top 5 high-signal decisions for RLM probing
    assistant_texts = [e for e in entries if e["type"] == "assistant_text"]
    all_decisions = _extract_decisions(assistant_texts)
    high_signal = [d for d in all_decisions if not d.startswith("[intent]")]
    facts["design_decisions"] = high_signal[:5]

    # 2. Error patterns — classified Bash/Edit failures for cross-session learning
    error_patterns = _classify_error_patterns(entries)
    if error_patterns:
        facts["error_patterns"] = error_patterns

    # 2.5. Success patterns — Edit/Write→Bash success sequences for cross-session learning
    success_patterns = _extract_success_patterns(entries)
    if success_patterns:
        facts["success_patterns"] = success_patterns

    # 3. pACS min-score — SOT에서 추출 (있는 경우, read-only)
    pacs_min = _extract_pacs_from_sot(project_dir)
    if pacs_min is not None:
        facts["pacs_min"] = pacs_min

    # 4. ULW mode detection — tag session for RLM cross-session queries
    ulw_state = detect_ulw_mode(entries)
    if ulw_state:
        facts["ulw_active"] = True

    # 5. FIX-H1: Team work summaries — archive to KI for RLM persistence
    # completed_summaries in snapshot IMMORTAL can be lost during Phase 6-7 compression.
    # Archiving to KI ensures cross-session team coordination history survives.
    # ETERNAL: These fields are preserved in quarterly archives even after rotation.
    team_summaries = _extract_team_summaries(project_dir)
    if team_summaries:
        facts["team_summaries"] = team_summaries

    # 6. Abductive Diagnosis patterns — archive to KI for cross-session learning
    # ETERNAL: These fields are preserved in quarterly archives even after rotation.
    diagnosis_patterns = _extract_diagnosis_patterns(project_dir)
    if diagnosis_patterns:
        facts["diagnosis_patterns"] = diagnosis_patterns

    # 7. H3: Hypothesis Graveyard — tried-and-failed approaches for cross-session learning
    rejected_hypotheses = _extract_hypothesis_graveyard(entries)
    if rejected_hypotheses:
        facts["rejected_hypotheses"] = rejected_hypotheses

    # 8. Phase 1-A: Thesis pending gates + blocked steps (session continuity markers)
    # Read thesis SOT (session.json) directly — no import from checklist_manager
    # to avoid circular dependency. Lightweight JSON read, read-only.
    thesis_continuity = _extract_thesis_continuity(project_dir)
    if thesis_continuity:
        facts["thesis_continuity"] = thesis_continuity

    # 9. Phase 1-B: Session type classification (deterministic)
    session_type = _classify_session_type(user_task, last_instruction, phase)
    if session_type:
        facts["session_type"] = session_type

    # R-12: gate_results — cross-session gate pass/fail for trend detection
    # Always extract gate results regardless of thesis_continuity status.
    # Completed project gates are valuable for cross-session trend analysis.
    gate_snapshot = _extract_gate_results_snapshot(project_dir)
    if gate_snapshot:
        facts["gate_results"] = gate_snapshot

    # CM-3: thesis_step — thesis current_step at archive time for proximity scoring.
    # Scalar int (not a range). Used by _retrieve_relevant_sessions() for step boost.
    # Reads thesis-output/*/session.json (same pattern as _extract_thesis_continuity).
    # P1 Compliance: Deterministic JSON read, non-blocking, read-only SOT access.
    _thesis_step = _extract_thesis_step_at_archive(project_dir)
    if _thesis_step is not None:
        facts["thesis_step"] = _thesis_step

    # B-3: invocation_number — active invocation at archive time for cross-session context.
    # Uses thesis_step to compute which invocation block is active (deterministic).
    # P1 Compliance: Deterministic computation from _INVOCATION_PLAN in query_step.py.
    if _thesis_step is not None:
        try:
            scripts_dir = os.path.join(os.path.dirname(__file__))
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from query_step import get_invocation_plan
            plan = get_invocation_plan(_thesis_step)
            # Prefer in_progress invocation; fallback to last completed
            in_progress = [p for p in plan if p["status"] == "in_progress"]
            if in_progress:
                facts["invocation_number"] = in_progress[0]["invocation"]
            else:
                completed = [p for p in plan if p["status"] == "completed"]
                if completed:
                    facts["invocation_number"] = completed[-1]["invocation"]
        except (ImportError, ModuleNotFoundError, AttributeError):
            pass  # Non-blocking: query_step.py may not exist or lack the function

    # B-2: hitl_decisions — cross-session HITL approval/rejection history
    # Enables Orchestrator to predict which checkpoints need human review.
    try:
        thesis_root = os.path.join(project_dir, "thesis-output")
        if os.path.isdir(thesis_root):
            for proj_name in sorted(os.listdir(thesis_root)):
                sot_path = os.path.join(thesis_root, proj_name, "session.json")
                if not os.path.isfile(sot_path):
                    continue
                try:
                    with open(sot_path, "r", encoding="utf-8") as f:
                        sot_data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                hitl_cp = sot_data.get("hitl_checkpoints", {})
                if isinstance(hitl_cp, dict) and hitl_cp:
                    hitl_summary = {}
                    for hname, hdata in hitl_cp.items():
                        if isinstance(hdata, dict):
                            hitl_summary[hname] = hdata.get("status", "pending")
                        else:
                            hitl_summary[hname] = str(hdata)
                    if hitl_summary:
                        facts["hitl_decisions"] = hitl_summary
                    break
    except Exception:
        pass  # Non-blocking

    # QO-5: Quality context fields for cross-session optimization
    # QO-5a: Previous section outputs — titles + word counts from thesis outputs
    try:
        thesis_root = os.path.join(project_dir, "thesis-output")
        if os.path.isdir(thesis_root):
            for proj_name in sorted(os.listdir(thesis_root)):
                sot_file = os.path.join(thesis_root, proj_name, "session.json")
                if not os.path.isfile(sot_file):
                    continue
                try:
                    with open(sot_file, "r", encoding="utf-8") as f:
                        sot_data_qo = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                outputs_qo = sot_data_qo.get("outputs", {})
                if not isinstance(outputs_qo, dict):
                    continue
                prev_outputs: list[dict] = []
                proj_root = os.path.join(thesis_root, proj_name)
                for key, val in sorted(outputs_qo.items()):
                    if not key.startswith("step-") or key.endswith("-ko"):
                        continue
                    try:
                        step_n = int(key.replace("step-", ""))
                    except (ValueError, TypeError):
                        continue
                    file_path_qo = val if isinstance(val, str) else ""
                    if file_path_qo:
                        full = os.path.join(proj_root, file_path_qo)
                        if os.path.exists(full):
                            try:
                                # H-2 Fix: Read full file for accurate word count.
                                # Previous 2KB cap severely undercounted large sections.
                                with open(full, "r", encoding="utf-8") as f:
                                    full_text = f.read()
                                wc = len(full_text.split())
                                # Heading extraction uses first 2KB only
                                title = ""
                                for ln in full_text[:2000].split("\n"):
                                    if ln.strip().startswith("## "):
                                        title = ln.strip()[3:].strip()[:60]
                                        break
                                prev_outputs.append({
                                    "step": step_n,
                                    "title": title,
                                    "words": wc,
                                })
                            except (IOError, OSError):
                                continue
                if prev_outputs:
                    facts["previous_section_outputs"] = prev_outputs[-10:]  # Last 10
                    # QO-5c: Word count trend
                    facts["word_count_trend"] = [
                        {"step": p["step"], "words": p["words"]}
                        for p in prev_outputs[-10:]
                    ]
                break  # First active project only
    except Exception:
        pass  # Non-blocking

    # QO-5b: Review feedback summary — from latest review log
    # H-3 Fix: Use parse_review_verdict() (same file) for deterministic extraction.
    # Previous raw regex ("VERDICT" in line) caused false positives on descriptive text.
    try:
        thesis_root_rv = os.path.join(project_dir, "thesis-output")
        if os.path.isdir(thesis_root_rv):
            for proj_name_rv in sorted(os.listdir(thesis_root_rv)):
                review_dir = os.path.join(thesis_root_rv, proj_name_rv, "review-logs")
                if not os.path.isdir(review_dir):
                    continue
                review_files = [
                    os.path.join(review_dir, f) for f in os.listdir(review_dir)
                    if f.endswith(".md")
                ]
                if not review_files:
                    continue
                review_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
                latest_review = review_files[0]
                try:
                    verdict_data = parse_review_verdict(latest_review)
                    rv_name = os.path.basename(latest_review)
                    verdict_str = verdict_data.get("verdict") or "UNKNOWN"
                    critical_n = verdict_data.get("critical_count", 0)
                    warning_n = verdict_data.get("warning_count", 0)
                    suggestion_n = verdict_data.get("suggestion_count", 0)
                    parts: list[str] = [rv_name, f"Verdict: {verdict_str}"]
                    if critical_n:
                        parts.append(f"Critical: {critical_n}")
                    if warning_n:
                        parts.append(f"Warnings: {warning_n}")
                    if suggestion_n:
                        parts.append(f"Suggestions: {suggestion_n}")
                    facts["review_feedback_summary"] = " | ".join(parts)
                except (IOError, OSError):
                    pass
                break  # First project only
    except Exception:
        pass  # Non-blocking

    # Mark ETERNAL fields for archival protection
    facts["_eternal_fields"] = ["team_summaries", "diagnosis_patterns", "design_decisions"]

    return facts


def replace_or_append_session_facts(ki_path, facts):
    """Append session facts to knowledge-index.jsonl with session_id dedup.

    If an entry with the same session_id already exists, replaces it
    (later saves have more complete data — e.g., sessionend after threshold).

    A-1: Reads under shared lock, writes via atomic temp→rename under exclusive lock.
         Even if the process crashes mid-write, the original file is never corrupted.
    A-2: Empty/missing session_id skips dedup (appends as new unique entry).
    A-3: Empty session_id triggers UUID fallback to prevent unbounded dedup bypass.

    P1 Compliance: All operations are deterministic (JSON read/filter/write).
    SOT Compliance: Only called from save_context.py and _trigger_proactive_save.
    """
    session_id = facts.get("session_id", "")

    # A-3: Empty session_id fallback — generate UUID to enable dedup on retry
    if not session_id or session_id == "unknown":
        import uuid
        session_id = f"auto-{uuid.uuid4().hex[:12]}"
        facts["session_id"] = session_id

    # P1 Schema Validation: Ensure RLM-critical keys exist before write
    facts = _validate_session_facts(facts)

    parent_dir = os.path.dirname(ki_path)
    os.makedirs(parent_dir, exist_ok=True)

    # Use a dedicated lock file to separate read/write locking from the data file.
    # This avoids the truncate-then-write vulnerability entirely.
    lock_path = ki_path + ".lock"

    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            # Read existing entries (file may not exist yet)
            lines = []
            if os.path.exists(ki_path):
                try:
                    with open(ki_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                except Exception:
                    pass

            # Filter out existing entry with same session_id (dedup)
            kept = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                    if entry.get("session_id") == session_id:
                        continue  # Remove old entry — will be replaced
                except json.JSONDecodeError:
                    kept.append(stripped + "\n")
                    continue
                kept.append(stripped + "\n")

            # Append new entry
            kept.append(json.dumps(facts, ensure_ascii=False) + "\n")

            # A-1: Atomic write — temp file + rename. If crash happens,
            # either old file or new file exists, never a half-written state.
            atomic_write(ki_path, "".join(kept))
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    except Exception:
        # Non-blocking fallback: append-only (no dedup, but no data loss)
        try:
            with open(ki_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(facts, ensure_ascii=False) + "\n")
        except Exception:
            pass


def cleanup_knowledge_index(snapshot_dir):
    """Rotate knowledge-index.jsonl with tiered archival.

    Instead of discarding old entries permanently, entries beyond
    MAX_KNOWLEDGE_INDEX_ENTRIES are compressed into quarterly summaries
    in knowledge-archive-quarterly.jsonl. This preserves long-term
    cross-session learning patterns (team coordination, error resolution,
    diagnosis patterns) that would otherwise be lost.

    Tiered archival strategy:
      - Active index: most recent 200 entries (full detail)
      - Quarterly archive: older entries compressed by quarter
        (aggregated error_patterns, design_decisions, team_summaries)
    """
    ki_path = os.path.join(snapshot_dir, "knowledge-index.jsonl")
    if not os.path.exists(ki_path):
        return

    try:
        lines = []
        with open(ki_path, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]

        if len(lines) <= MAX_KNOWLEDGE_INDEX_ENTRIES:
            return

        # Split: overflow (oldest) → archive, keep (newest) → active
        overflow = lines[:-MAX_KNOWLEDGE_INDEX_ENTRIES]
        trimmed = lines[-MAX_KNOWLEDGE_INDEX_ENTRIES:]

        # Archive overflow entries as quarterly summaries
        _archive_to_quarterly(snapshot_dir, overflow)

        # Write trimmed active index
        atomic_write(ki_path, "".join(trimmed))
    except Exception:
        pass


def _archive_to_quarterly(snapshot_dir, overflow_lines):
    """Compress overflow entries into quarterly summaries.

    Groups entries by quarter (YYYY-Q#), aggregates key fields:
    error_patterns, design_decisions, team_summaries, diagnosis_patterns.
    Appends to knowledge-archive-quarterly.jsonl (never overwrites).
    """
    import collections

    archive_path = os.path.join(snapshot_dir, "knowledge-archive-quarterly.jsonl")
    quarters = collections.defaultdict(lambda: {
        "session_count": 0,
        "error_patterns": collections.Counter(),
        "design_decisions": [],
        "team_summaries": [],
        "diagnosis_patterns": [],
        "modified_files": collections.Counter(),
        "tools_used": collections.Counter(),
    })

    for line in overflow_lines:
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        # Determine quarter from timestamp
        ts = entry.get("timestamp", "")
        if len(ts) >= 7:
            year_month = ts[:7]  # "2026-03"
            try:
                month = int(year_month.split("-")[1])
                quarter = (month - 1) // 3 + 1
                qkey = f"{year_month[:4]}-Q{quarter}"
            except (ValueError, IndexError):
                qkey = "unknown"
        else:
            qkey = "unknown"

        q = quarters[qkey]
        q["session_count"] += 1

        # Aggregate error patterns
        for ep in entry.get("error_patterns", []):
            if isinstance(ep, dict):
                q["error_patterns"][ep.get("type", "unknown")] += ep.get("count", 1)
            elif isinstance(ep, str):
                q["error_patterns"][ep] += 1

        # Collect design decisions (deduplicate by content)
        for dd in entry.get("design_decisions", []):
            if dd and dd not in q["design_decisions"]:
                q["design_decisions"].append(dd)

        # Collect team summaries (handles both list and dict formats)
        ts_data = entry.get("team_summaries", [])
        if isinstance(ts_data, dict):
            ts_data = list(ts_data.values())
        for ts_entry in ts_data:
            if ts_entry:
                q["team_summaries"].append(ts_entry)

        # Collect diagnosis patterns
        for dp in entry.get("diagnosis_patterns", []):
            if dp and dp not in q["diagnosis_patterns"]:
                q["diagnosis_patterns"].append(dp)

        # Aggregate files and tools
        for f in entry.get("modified_files", []):
            q["modified_files"][f] += 1
        tu = entry.get("tools_used", {})
        if isinstance(tu, dict):
            for t, count in tu.items():
                q["tools_used"][t] += count if isinstance(count, int) else 1
        else:
            for t in tu:
                q["tools_used"][t] += 1

    # Write quarterly summaries (append mode)
    try:
        with open(archive_path, "a", encoding="utf-8") as f:
            for qkey, q in sorted(quarters.items()):
                summary = {
                    "quarter": qkey,
                    "session_count": q["session_count"],
                    "error_patterns_aggregated": dict(q["error_patterns"]),
                    "design_decisions": q["design_decisions"][:20],
                    "team_summaries": q["team_summaries"][:10],
                    "diagnosis_patterns": q["diagnosis_patterns"][:10],
                    "top_modified_files": dict(q["modified_files"].most_common(20)),
                    "top_tools": dict(q["tools_used"].most_common(10)),
                    "archived_at": datetime.now(timezone.utc).isoformat(),
                }
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    except Exception:
        pass  # Non-blocking — archival is supplementary


def cleanup_session_archives(snapshot_dir):
    """Rotate session archives to keep MAX_SESSION_ARCHIVES files.

    Keeps most recent by modification time.
    """
    sessions_dir = os.path.join(snapshot_dir, "sessions")
    if not os.path.isdir(sessions_dir):
        return

    try:
        files = []
        for f in os.listdir(sessions_dir):
            if f.endswith(".md"):
                fpath = os.path.join(sessions_dir, f)
                files.append((fpath, os.path.getmtime(fpath)))

        if len(files) <= MAX_SESSION_ARCHIVES:
            return

        # Sort by mtime, newest first — remove oldest
        files.sort(key=lambda x: x[1], reverse=True)
        for fpath, _ in files[MAX_SESSION_ARCHIVES:]:
            try:
                os.unlink(fpath)
            except OSError:
                pass
    except Exception:
        pass


# =============================================================================
# Quality Gate State Extraction (Context Memory Optimization)
# =============================================================================


def _extract_quality_gate_state(project_dir):
    """Extract latest Quality Gate state for IMMORTAL snapshot preservation.

    Scans pacs-logs/, review-logs/, verification-logs/ for the most recent
    step's quality gate results. Provides session recovery context when
    a session dies during Verification retry, pACS RED rework, or Review FAIL.

    P1 Compliance: Filesystem + regex only.
    SOT Compliance: Read-only access to log directories.

    Returns: list of markdown lines (empty if no gate logs exist).
    """
    lines = []

    # Find the highest step number across all gate log directories
    max_step = 0
    gate_dirs = {
        "pacs": os.path.join(project_dir, "pacs-logs"),
        "review": os.path.join(project_dir, "review-logs"),
        "verify": os.path.join(project_dir, "verification-logs"),
    }
    _step_re = re.compile(r"step-(\d+)")

    for gate_type, gate_dir in gate_dirs.items():
        if not os.path.isdir(gate_dir):
            continue
        try:
            for fname in os.listdir(gate_dir):
                m = _step_re.search(fname)
                if m:
                    step = int(m.group(1))
                    if step > max_step:
                        max_step = step
        except OSError:
            continue

    # In-progress dialogue check — MUST run before early return so it fires
    # even when max_step==0 (no pacs/review/verify logs yet, but dialogue started).
    # RLM gap fix: when context resets mid-dialogue, inject current round state
    # as a POINTER so Orchestrator knows to read session.json.dialogue_state.
    _session_json_path = os.path.join(project_dir, "session.json")
    if os.path.exists(_session_json_path):
        try:
            with open(_session_json_path, "r", encoding="utf-8") as _f:
                _session_data = json.load(_f)
            _ds = _session_data.get("dialogue_state")
            if isinstance(_ds, dict) and _ds.get("status") == "in_progress":
                _rounds_used = _ds.get("rounds_used", 0)
                _max_rounds = _ds.get("max_rounds", "?")
                _domain = _ds.get("domain", "?")
                _step_num = _ds.get("step", max_step)
                _last_verdict = "none"
                _history = _ds.get("round_history", [])
                if isinstance(_history, list) and _history:
                    _last_entry = _history[-1]
                    if isinstance(_last_entry, dict):
                        _last_verdict = _last_entry.get("verdict", "none")
                lines.append(
                    f"- **Dialogue [IN-PROGRESS]**: step={_step_num}, "
                    f"round={_rounds_used}/{_max_rounds}, domain={_domain}, "
                    f"last_verdict={_last_verdict} — "
                    f"Read session.json.dialogue_state for full state"
                )
        except Exception:
            pass

    if max_step == 0:
        return lines

    lines.append(f"최신 검증 단계: **Step {max_step}**")

    # pACS score for the latest step
    pacs_path = os.path.join(
        project_dir, "pacs-logs", f"step-{max_step}-pacs.md"
    )
    if os.path.exists(pacs_path):
        try:
            with open(pacs_path, "r", encoding="utf-8") as f:
                pacs_content = f.read(2000)
            if not pacs_content.strip():
                raise ValueError("empty pACS file")
            # Extract pACS score
            pacs_match = re.search(
                r"pACS\s*=.*?=\s*(\d{1,3})|pACS\s*=\s*(\d{1,3})",
                pacs_content, re.IGNORECASE,
            )
            if pacs_match:
                score = pacs_match.group(1) or pacs_match.group(2)
                lines.append(f"- **pACS**: {score}")
            # Extract weak dimension
            weak_match = re.search(
                r"(?:weak|약점)\s*(?:dimension|차원)\s*[:=]\s*([FCL])",
                pacs_content, re.IGNORECASE,
            )
            if weak_match:
                lines.append(f"- **약점 차원**: {weak_match.group(1)}")
            # Extract pre-mortem first line
            pm_match = re.search(
                r"(?:Pre-mortem|사전 부검).*?\n(.*?)(?:\n|$)",
                pacs_content, re.IGNORECASE,
            )
            if pm_match:
                pm_line = pm_match.group(1).strip()[:200]
                if pm_line:
                    lines.append(f"- **Pre-mortem**: {pm_line}")
        except Exception:
            pass

    # Review verdict for the latest step
    review_path = os.path.join(
        project_dir, "review-logs", f"step-{max_step}-review.md"
    )
    if os.path.exists(review_path):
        review_data = parse_review_verdict(review_path)
        verdict = review_data.get("verdict", "N/A")
        critical = review_data.get("critical_count", 0)
        warning = review_data.get("warning_count", 0)
        reviewer_pacs = review_data.get("reviewer_pacs")
        lines.append(
            f"- **Review**: {verdict} "
            f"(Critical: {critical}, Warning: {warning})"
        )
        if reviewer_pacs:
            lines.append(f"- **Reviewer pACS**: {reviewer_pacs}")

    # Verification status for the latest step
    verify_path = os.path.join(
        project_dir, "verification-logs", f"step-{max_step}-verify.md"
    )
    if os.path.exists(verify_path):
        try:
            with open(verify_path, "r", encoding="utf-8") as f:
                verify_content = f.read(2000)
            pass_count = len(re.findall(
                r"\bPASS\b", verify_content, re.IGNORECASE
            ))
            fail_count = len(re.findall(
                r"\bFAIL\b", verify_content, re.IGNORECASE
            ))
            lines.append(
                f"- **Verification**: PASS {pass_count}건, FAIL {fail_count}건"
            )
        except Exception:
            pass

    # Diagnosis status for the latest step (if diagnosis-logs/ exists)
    diag_dir = os.path.join(project_dir, "diagnosis-logs")
    if os.path.isdir(diag_dir):
        try:
            diag_files = [
                f for f in os.listdir(diag_dir)
                if f.startswith(f"step-{max_step}-") and f.endswith(".md")
            ]
            if diag_files:
                latest_diag = sorted(diag_files)[-1]
                diag_path = os.path.join(diag_dir, latest_diag)
                with open(diag_path, "r", encoding="utf-8") as f:
                    diag_content = f.read(2000)
                selected = _DIAG_SELECTED_RE.search(diag_content)
                gate_match = _DIAG_GATE_RE.search(diag_content)
                diag_gate = gate_match.group(1) if gate_match else "?"
                diag_hyp = selected.group(1).strip() if selected else "?"
                lines.append(
                    f"- **Diagnosis**: gate={diag_gate}, hypothesis={diag_hyp}"
                )
        except Exception:
            pass

    # H4: Retry trajectory tracking — scan all attempts for the latest step
    # Collect pACS scores across retry attempts (step-N-pacs.md, step-N-retry-1-pacs.md, etc.)
    pacs_dir = os.path.join(project_dir, "pacs-logs")
    if os.path.isdir(pacs_dir):
        try:
            score_history = []
            retry_pattern = re.compile(
                rf"step-{max_step}(?:-retry-(\d+))?-pacs\.md$"
            )
            pacs_files = []
            for fname in os.listdir(pacs_dir):
                m = retry_pattern.match(fname)
                if m:
                    retry_num = int(m.group(1)) if m.group(1) else 0
                    pacs_files.append((retry_num, fname))
            pacs_files.sort()
            for _, fname in pacs_files:
                fpath = os.path.join(pacs_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read(2000)
                    sm = re.search(
                        r"pACS\s*=.*?=\s*(\d{1,3})|pACS\s*=\s*(\d{1,3})",
                        content, re.IGNORECASE,
                    )
                    if sm:
                        score_history.append(int(sm.group(1) or sm.group(2)))
                except Exception:
                    continue

            attempts = len(pacs_files)
            if attempts > 0:
                # Determine trend from last 3 scores
                trend = "unknown"
                if len(score_history) >= 3:
                    last3 = score_history[-3:]
                    spread = max(last3) - min(last3)
                    if spread <= 5:
                        trend = "plateauing"
                    elif last3[-1] > last3[0]:
                        trend = "improving"
                    else:
                        trend = "degrading"
                elif len(score_history) == 2:
                    if score_history[-1] > score_history[0]:
                        trend = "improving"
                    elif score_history[-1] < score_history[0]:
                        trend = "degrading"
                    else:
                        trend = "plateauing"
                elif len(score_history) == 1:
                    trend = "initial"

                lines.append(
                    f"- **Retry Trajectory**: attempts={attempts}, "
                    f"scores={score_history[-5:]}, trend={trend}"
                )
        except Exception:
            pass

    # Dialogue state for the latest step (if dialogue-logs/ exists and summary present)
    # Priority 1: completed dialogue → read step-N-summary.md
    # Priority 2: in-progress dialogue → read session.json.dialogue_state (RLM gap fix)
    # This ensures context recovery mid-dialogue restores current round/domain/feedback.
    dialogue_summary_path = os.path.join(
        project_dir, "dialogue-logs", f"step-{max_step}-summary.md"
    )
    if os.path.exists(dialogue_summary_path):
        try:
            with open(dialogue_summary_path, "r", encoding="utf-8") as f:
                dlg_content = f.read(2000)
            outcome_match = re.search(
                r"Outcome\s*:\s*(consensus|escalated)", dlg_content, re.IGNORECASE
            )
            rounds_match = re.search(r"Rounds\s+Used\s*:\s*(\d+)", dlg_content, re.IGNORECASE)
            if outcome_match:
                dlg_outcome = outcome_match.group(1)
                dlg_rounds = rounds_match.group(1) if rounds_match else "?"
                lines.append(
                    f"- **Dialogue**: outcome={dlg_outcome}, rounds={dlg_rounds}"
                )
        except Exception:
            pass
    # Note: in-progress dialogue is handled before the max_step==0 early return above.
    # No else: branch needed here — avoids duplicate injection when max_step>0.

    return lines


# =============================================================================
# Predictive Debugging: Risk Score Aggregation (P1 — Deterministic)
# =============================================================================

def aggregate_risk_scores(ki_path, project_dir):
    """Aggregate per-file risk scores from knowledge-index.jsonl.

    P1 Compliance: All operations are deterministic arithmetic.
    No semantic inference — pure counting, weighting, and decay.

    Called by: restore_context.py at SessionStart (once per session).
    Output: dict suitable for JSON serialization to risk-scores.json.

    Data flow:
      knowledge-index.jsonl → read entries → extract error_patterns
      → per-file error counting → weight application → recency decay
      → resolution rate calculation → validate → return
    """
    # Read all knowledge-index entries
    entries = []
    if not ki_path or not os.path.exists(ki_path):
        return _empty_risk_data(project_dir)

    try:
        with open(ki_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        return _empty_risk_data(project_dir)

    if len(entries) < _RISK_MIN_SESSIONS:
        return _empty_risk_data(project_dir, data_sessions=len(entries))

    # Per-file error accumulation
    # Key: relative path → {error_types: {type: count}, total_weighted: float,
    #                        resolved_count: int, total_count: int, last_error_date: str}
    file_risks = {}
    now_ts = time.time()

    for entry in entries:
        error_patterns = entry.get("error_patterns", [])
        if not isinstance(error_patterns, list):
            continue

        # Parse entry timestamp for recency decay
        entry_ts = entry.get("timestamp", "")
        entry_age_days = _timestamp_to_age_days(entry_ts, now_ts)

        # Determine recency weight
        recency_weight = _RECENCY_DECAY_DAYS[-1][1]  # default: oldest bracket
        for max_days, weight in _RECENCY_DECAY_DAYS:
            if entry_age_days <= max_days:
                recency_weight = weight
                break

        # Get modified files from this session (for file↔error association)
        modified_files = entry.get("modified_files", [])

        for ep in error_patterns:
            if not isinstance(ep, dict):
                continue
            error_type = ep.get("type", "unknown")
            error_file = ep.get("file", "")
            resolution = ep.get("resolution")
            has_resolution = isinstance(resolution, dict) and bool(resolution)

            # Determine which file(s) to attribute the error to
            # Priority: error-specific file > session modified files
            target_files = []
            if error_file:
                target_files = [error_file]
            else:
                # No specific file — attribute to all modified files
                target_files = [os.path.basename(f) for f in modified_files[:5]]

            for tf in target_files:
                # Normalize to relative path
                rel_path = _normalize_to_relative(tf, project_dir, modified_files)
                if not rel_path:
                    continue

                if rel_path not in file_risks:
                    file_risks[rel_path] = {
                        "error_types": {},
                        "total_weighted": 0.0,
                        "resolved_count": 0,
                        "total_count": 0,
                        "last_error_date": "",
                    }

                fr = file_risks[rel_path]
                # Apply type weight × recency weight
                type_weight = _RISK_WEIGHTS.get(error_type, 0.7)
                weighted_score = type_weight * recency_weight
                fr["total_weighted"] += weighted_score
                fr["error_types"][error_type] = fr["error_types"].get(error_type, 0) + 1
                fr["total_count"] += 1
                if has_resolution:
                    fr["resolved_count"] += 1

                # Track most recent error date
                entry_date = entry_ts[:10] if len(entry_ts) >= 10 else ""
                if entry_date > fr["last_error_date"]:
                    fr["last_error_date"] = entry_date

    # P1-FIX: Merge entries with same basename but different paths
    # (bare names like "_context_lib.py" vs relative ".claude/hooks/scripts/_context_lib.py")
    # Keep the longest (most specific) path as canonical key, sum scores.
    basename_groups = {}
    for rel_path, fr in file_risks.items():
        bname = os.path.basename(rel_path)
        if bname not in basename_groups:
            basename_groups[bname] = []
        basename_groups[bname].append((rel_path, fr))

    merged_risks = {}
    for bname, group in basename_groups.items():
        if len(group) == 1:
            merged_risks[group[0][0]] = group[0][1]
        else:
            # Pick longest path as canonical (most specific)
            canonical_path = max(group, key=lambda x: len(x[0]))[0]
            merged = {
                "error_types": {},
                "total_weighted": 0.0,
                "resolved_count": 0,
                "total_count": 0,
                "last_error_date": "",
            }
            for _, fr in group:
                merged["total_weighted"] += fr["total_weighted"]
                merged["total_count"] += fr["total_count"]
                merged["resolved_count"] += fr["resolved_count"]
                for etype, cnt in fr["error_types"].items():
                    merged["error_types"][etype] = (
                        merged["error_types"].get(etype, 0) + cnt
                    )
                if fr["last_error_date"] > merged["last_error_date"]:
                    merged["last_error_date"] = fr["last_error_date"]
            merged_risks[canonical_path] = merged

    # Build output
    files_output = {}
    for rel_path, fr in merged_risks.items():
        resolution_rate = (
            fr["resolved_count"] / fr["total_count"]
            if fr["total_count"] > 0
            else 0.0
        )
        files_output[rel_path] = {
            "risk_score": round(fr["total_weighted"], 2),
            "error_count": fr["total_count"],
            "error_types": fr["error_types"],
            "last_error_session": fr["last_error_date"],
            "resolution_rate": round(resolution_rate, 2),
        }

    # Sort by risk_score descending for top_risk_files
    sorted_files = sorted(
        files_output.keys(),
        key=lambda k: files_output[k]["risk_score"],
        reverse=True,
    )
    top_risk = [
        f for f in sorted_files[:10]
        if files_output[f]["risk_score"] >= _RISK_SCORE_THRESHOLD
    ]

    risk_data = {
        "generated_at": datetime.now().isoformat(),
        "data_sessions": len(entries),
        "project_dir": project_dir,
        "risk_threshold": _RISK_SCORE_THRESHOLD,
        "files": files_output,
        "top_risk_files": top_risk,
    }

    # P1: Self-validation before return
    validation_warnings = validate_risk_scores(risk_data)
    if validation_warnings:
        risk_data["_validation_warnings"] = validation_warnings

    return risk_data


def validate_risk_scores(risk_data):
    """P1 Risk Score Validation (RS1-RS6).

    Deterministic schema enforcement for risk-scores.json.
    Follows the same pattern as validate_sot_schema (S1-S8),
    validate_review_output (R1-R5), validate_translation_output (T1-T7).

    Returns: list of warning strings (empty = all checks pass).
    """
    warnings = []

    if not isinstance(risk_data, dict):
        warnings.append("RS1 FAIL: risk_data is not a dict")
        return warnings

    # RS1: Required top-level keys
    required_keys = {"generated_at", "data_sessions", "files", "top_risk_files", "risk_threshold"}
    missing = required_keys - set(risk_data.keys())
    if missing:
        warnings.append(f"RS1 FAIL: Missing required keys: {missing}")

    # RS2: data_sessions is int >= 0
    ds = risk_data.get("data_sessions")
    if not isinstance(ds, int) or ds < 0:
        warnings.append(f"RS2 FAIL: data_sessions must be int >= 0, got {ds!r}")

    # RS3-RS5: Per-file validation
    files = risk_data.get("files", {})
    if not isinstance(files, dict):
        warnings.append("RS3 FAIL: files must be a dict")
    else:
        for fpath, fdata in files.items():
            if not isinstance(fdata, dict):
                warnings.append(f"RS3 FAIL: files[{fpath!r}] is not a dict")
                continue

            # RS3: risk_score is numeric >= 0
            score = fdata.get("risk_score")
            if not isinstance(score, (int, float)) or score < 0:
                warnings.append(
                    f"RS3 FAIL: files[{fpath!r}].risk_score must be "
                    f"numeric >= 0, got {score!r}"
                )

            # RS4: error_count >= sum(error_types.values())
            ec = fdata.get("error_count", 0)
            et = fdata.get("error_types", {})
            if isinstance(et, dict) and isinstance(ec, int):
                type_sum = sum(
                    v for v in et.values() if isinstance(v, (int, float))
                )
                if ec < type_sum:
                    warnings.append(
                        f"RS4 FAIL: files[{fpath!r}].error_count ({ec}) < "
                        f"sum(error_types) ({type_sum})"
                    )

            # RS5: resolution_rate is float, 0.0 <= rate <= 1.0
            rr = fdata.get("resolution_rate")
            if rr is not None:
                if not isinstance(rr, (int, float)) or rr < 0.0 or rr > 1.0:
                    warnings.append(
                        f"RS5 FAIL: files[{fpath!r}].resolution_rate must be "
                        f"0.0-1.0, got {rr!r}"
                    )

    # RS6: top_risk_files entries exist in files dict and sorted by risk_score desc
    top = risk_data.get("top_risk_files", [])
    if isinstance(top, list) and isinstance(files, dict):
        for tf in top:
            if tf not in files:
                warnings.append(
                    f"RS6 FAIL: top_risk_files entry {tf!r} not found in files"
                )
        # Check sort order
        scores = [
            files.get(tf, {}).get("risk_score", 0)
            for tf in top if tf in files
        ]
        if scores != sorted(scores, reverse=True):
            warnings.append(
                "RS6 FAIL: top_risk_files not sorted by risk_score desc"
            )

    return warnings


def _empty_risk_data(project_dir, data_sessions=0):
    """Return empty risk data structure (cold start / no data)."""
    return {
        "generated_at": datetime.now().isoformat(),
        "data_sessions": data_sessions,
        "project_dir": project_dir,
        "risk_threshold": _RISK_SCORE_THRESHOLD,
        "files": {},
        "top_risk_files": [],
    }


def _timestamp_to_age_days(ts_str, now_ts):
    """Convert ISO timestamp string to age in days.

    P1 Compliance: Deterministic datetime parsing.
    Returns float (days). Returns 365.0 on parse failure (conservative decay).
    """
    if not ts_str:
        return 365.0
    try:
        # Handle both "2026-02-20T15:30:00" and "2026-02-20T153000" formats
        dt = datetime.fromisoformat(
            ts_str.replace("Z", "+00:00") if ts_str.endswith("Z") else ts_str
        )
        age_seconds = now_ts - dt.timestamp()
        return max(0.0, age_seconds / 86400.0)
    except (ValueError, TypeError, OSError):
        return 365.0  # Conservative: treat unparseable as old


def _normalize_to_relative(filename, project_dir, modified_files):
    """Normalize a filename to project-relative path.

    Strategy:
      1. If filename is a bare name, find full path in modified_files
      2. If filename is absolute and under project_dir, make relative
      3. Otherwise return as-is (best effort)

    P1 Compliance: Deterministic string operations only.
    Returns: relative path string, or empty string on failure.
    """
    if not filename:
        return ""

    # Case 1: Bare filename (no path separator) — find in modified_files
    if not os.path.isabs(filename) and os.sep not in filename:
        for mf in modified_files:
            if os.path.basename(mf) == filename:
                if os.path.isabs(mf) and project_dir:
                    try:
                        return os.path.relpath(mf, project_dir)
                    except ValueError:
                        return mf
                return mf
        return filename  # Return bare filename as-is (best effort)

    # Case 2: Absolute path — make relative to project
    if os.path.isabs(filename) and project_dir:
        try:
            return os.path.relpath(filename, project_dir)
        except ValueError:
            return filename

    return filename


# =============================================================================
# Abductive Diagnosis Layer (P1: Deterministic Pre/Post Analysis)
# =============================================================================
# Inserts a 3-step diagnosis (P1 pre-evidence → LLM judgment → P1 post-validation)
# between quality gate FAIL and retry. Existing 4-layer QA is NOT modified.
# SOT Compliance: Read-only access to SOT, verification-logs/, pacs-logs/,
#                 review-logs/, diagnosis-logs/.
# P1 Compliance: All evidence gathering and validation is deterministic.


def diagnose_failure_context(project_dir, step, gate, sot_data=None):
    """Pre-analysis: Gather deterministic evidence bundle for a failed quality gate.

    Called by Orchestrator AFTER a gate FAIL and BEFORE retry.
    Returns a dict with structured evidence for LLM-based hypothesis selection.

    Args:
        project_dir: Project root path.
        step: Step number that failed.
        gate: One of 'verification', 'pacs', 'review'.
        sot_data: Optional pre-loaded SOT dict (avoids re-reading).

    Returns:
        dict with keys: step, gate, retry_history, upstream_evidence,
                        hypothesis_priority, fast_path, raw_evidence.
    """
    retry_history = _gather_retry_history(project_dir, step, gate)
    upstream_evidence = _gather_upstream_evidence(project_dir, step, sot_data)
    hypothesis_priority = _determine_hypothesis_priority(
        retry_history, upstream_evidence, gate
    )
    fast_path = _check_fast_path_eligibility(
        project_dir, step, gate, retry_history, sot_data=sot_data
    )
    raw_evidence = _gather_raw_evidence(project_dir, step, gate)

    return {
        "step": step,
        "gate": gate,
        "retry_history": retry_history,
        "upstream_evidence": upstream_evidence,
        "hypothesis_priority": hypothesis_priority,
        "fast_path": fast_path,
        "raw_evidence": raw_evidence,
    }


def _gather_retry_history(project_dir, step, gate):
    """Read retry counter and previous diagnosis logs for this step+gate.

    Returns:
        dict with keys: retries_used (int), max_retries (int),
                        previous_diagnoses (list of dicts).
    """
    # D-7: Retry limit constants must match validate_retry_budget.py
    # DEFAULT_MAX_RETRIES and ULW_MAX_RETRIES. Change both files together.
    _DEFAULT_MAX_RETRIES = 10
    _ULW_MAX_RETRIES = 15

    result = {
        "retries_used": 0,
        "max_retries": _DEFAULT_MAX_RETRIES,
        "previous_diagnoses": [],
    }

    # Read retry counter
    counter_dir = os.path.join(project_dir, f"{gate}-logs")
    counter_file = os.path.join(counter_dir, f".step-{step}-retry-count")
    if os.path.exists(counter_file):
        try:
            with open(counter_file, "r", encoding="utf-8") as f:
                result["retries_used"] = int(f.read().strip() or "0")
        except (ValueError, OSError):
            pass

    # Detect ULW mode for max_retries adjustment (10 → 15)
    # D-7: ULW detection pattern must match validate_retry_budget.py _ULW_SNAPSHOT_RE
    # and restore_context.py — all use "ULW 상태" section header presence.
    snapshot_path = os.path.join(
        project_dir, ".claude", "context-snapshots", "latest.md"
    )
    if os.path.exists(snapshot_path):
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                content = f.read(8000)  # First 8KB only
            if re.search(r"ULW 상태|Ultrawork Mode State", content):
                result["max_retries"] = _ULW_MAX_RETRIES
        except OSError:
            pass

    # Gather previous diagnosis logs
    diag_dir = os.path.join(project_dir, "diagnosis-logs")
    if os.path.isdir(diag_dir):
        try:
            for fname in sorted(os.listdir(diag_dir)):
                if fname.startswith(f"step-{step}-{gate}-") and fname.endswith(".md"):
                    fpath = os.path.join(diag_dir, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            content = f.read()
                        selected = _DIAG_SELECTED_RE.search(content)
                        result["previous_diagnoses"].append({
                            "file": fname,
                            "selected_hypothesis": selected.group(1).strip() if selected else "unknown",
                        })
                    except OSError:
                        pass
        except OSError:
            pass

    return result


def _gather_upstream_evidence(project_dir, step, sot_data=None):
    """Collect evidence from upstream step outputs referenced by SOT.

    Returns:
        dict with keys: upstream_outputs (list of {step, path, exists, size}),
                        sot_current_step (int), sot_status (str).
    """
    result = {
        "upstream_outputs": [],
        "sot_current_step": step,
        "sot_status": "unknown",
    }

    # Load SOT if not provided
    if sot_data is None:
        sot_data = {}
        try:
            import yaml
            for sp in sot_paths(project_dir):
                if os.path.exists(sp):
                    with open(sp, "r", encoding="utf-8") as f:
                        sot_data = yaml.safe_load(f) or {}
                    break
        except Exception:
            pass

    result["sot_current_step"] = sot_data.get("current_step", step)
    result["sot_status"] = sot_data.get("workflow_status", "unknown")

    # Gather upstream outputs (steps 1..step-1)
    # Guard: YAML `outputs: null` returns None, not {}
    outputs = sot_data.get("outputs") or {}
    for prev_step in range(1, step):
        key = f"step-{prev_step}"
        path_raw = outputs.get(key, "")
        if not path_raw:
            continue
        full_path = os.path.join(project_dir, path_raw)
        result["upstream_outputs"].append({
            "step": prev_step,
            "path": path_raw,
            "exists": os.path.exists(full_path),
            "size": os.path.getsize(full_path) if os.path.exists(full_path) else 0,
        })

    return result


def _determine_hypothesis_priority(retry_history, upstream_evidence, gate):
    """Rule-based hypothesis prioritization based on available evidence.

    Four hypothesis categories (H1, H2, H3, H4):
        H1: Upstream data quality (missing/thin upstream outputs)
        H2: Current step execution gap (most common)
        H3: Criteria interpretation error (rare)
        H4: Capability gap — missing tool, script, or infrastructure

    Returns:
        list of dicts with keys: id (str), label (str), priority (int 1-3),
                                  reason (str).
    """
    hypotheses = []

    # H1: Upstream data quality — check if any upstream output is missing/thin
    thin_upstreams = []
    for uo in upstream_evidence.get("upstream_outputs", []):
        if not uo.get("exists"):
            thin_upstreams.append(f"step-{uo['step']} missing")
        elif uo.get("size", 0) < MIN_OUTPUT_SIZE:
            thin_upstreams.append(f"step-{uo['step']} thin ({uo['size']}B)")

    h1_priority = 1 if thin_upstreams else 3
    hypotheses.append({
        "id": "H1",
        "label": "Upstream data quality issue",
        "priority": h1_priority,
        "reason": "; ".join(thin_upstreams) if thin_upstreams else "All upstream outputs present and adequate",
    })

    # H2: Current step execution gap — most common, default high priority
    prev_diag = retry_history.get("previous_diagnoses", [])
    h2_priority = 2 if prev_diag else 1
    # If previous diagnosis already selected H2, lower priority (try different hypothesis)
    if prev_diag and any(
        d.get("selected_hypothesis", "").startswith("H2") or
        "execution" in d.get("selected_hypothesis", "").lower()
        for d in prev_diag
    ):
        h2_priority = 2

    hypotheses.append({
        "id": "H2",
        "label": "Current step execution gap",
        "priority": h2_priority,
        "reason": f"{len(prev_diag)} previous diagnosis(es)" if prev_diag else "First attempt",
    })

    # H3: Criteria interpretation error — higher priority for review gate
    # Also elevated for verification gate when V1d failures suggest hallucinated evidence
    h3_priority = 3
    if gate == "review":
        h3_priority = 2
    elif gate == "verification":
        # Check if raw evidence contains V1d failures (hallucinated evidence pattern)
        v1d_hint = any(
            "V1d" in d.get("selected_hypothesis", "") or
            "evidence" in d.get("selected_hypothesis", "").lower()
            for d in prev_diag
        )
        if v1d_hint:
            h3_priority = 1  # Hallucination pattern → criteria re-examination critical
    hypotheses.append({
        "id": "H3",
        "label": "Criteria interpretation error",
        "priority": h3_priority,
        "reason": (
            "Review gate benefits from criteria re-examination" if gate == "review"
            else "Verification gate with evidence quality issues" if h3_priority == 1
            else "Low prior probability"
        ),
    })

    # H4: Capability gap — missing tool, script, or infrastructure
    # Elevated priority when: (a) repeated retries with same H2, (b) error
    # patterns suggest missing commands/tools. OpenAI harness pattern:
    # "build the missing capability rather than retrying manually."
    h4_priority = 3  # default: low
    h2_repeats = sum(
        1 for d in prev_diag
        if d.get("selected_hypothesis", "").startswith("H2")
    )
    if h2_repeats >= 2:
        # Two H2 attempts failed → likely not an execution gap but a missing capability
        h4_priority = 1
    hypotheses.append({
        "id": "H4",
        "label": "Capability gap — missing tool, script, or infrastructure",
        "priority": h4_priority,
        "reason": (
            f"H2 selected {h2_repeats} times without resolution — "
            "consider building missing capability"
            if h2_repeats >= 2
            else "Low prior probability — check after H2 exhausted"
        ),
    })

    # Sort by priority (1 = highest)
    hypotheses.sort(key=lambda h: h["priority"])
    return hypotheses


def _check_fast_path_eligibility(project_dir, step, gate, retry_history,
                                 sot_data=None):
    """Deterministic fast-path checks (FP1-FP3) that skip LLM diagnosis.

    FP1: Missing output file — diagnosis is trivially 'file not generated'.
    FP2: Empty/near-empty output — diagnosis is 'incomplete generation'.
    FP3: Identical retry — same hypothesis selected twice without change.

    Args:
        sot_data: Optional pre-loaded SOT dict (avoids redundant I/O).

    Returns:
        dict with keys: eligible (bool), reason (str), fp_id (str or None).
    """
    result = {"eligible": False, "reason": "", "fp_id": None}

    # FP1: Missing output file for current step
    try:
        if sot_data is None:
            import yaml
            sot_data = {}
            for sp in sot_paths(project_dir):
                if os.path.exists(sp):
                    with open(sp, "r", encoding="utf-8") as f:
                        sot_data = yaml.safe_load(f) or {}
                    break
        # Guard: YAML `outputs: null` returns None, not {}
        outputs = sot_data.get("outputs") or {}
        step_key = f"step-{step}"
        output_path_raw = outputs.get(step_key, "")
        if output_path_raw:
            full_path = os.path.join(project_dir, output_path_raw)
            if not os.path.exists(full_path):
                result["eligible"] = True
                result["reason"] = f"FP1: Output file missing — {output_path_raw}"
                result["fp_id"] = "FP1"
                return result
            # FP2: Empty/near-empty output
            fsize = os.path.getsize(full_path)
            if fsize < MIN_OUTPUT_SIZE:
                result["eligible"] = True
                result["reason"] = f"FP2: Output too small ({fsize}B < {MIN_OUTPUT_SIZE}B)"
                result["fp_id"] = "FP2"
                return result
    except Exception:
        pass

    # FP3: Identical retry — same hypothesis selected in 2+ previous diagnoses
    prev_diag = retry_history.get("previous_diagnoses", [])
    if len(prev_diag) >= 2:
        selected = [d.get("selected_hypothesis", "") for d in prev_diag[-2:]]
        if selected[0] and selected[0] == selected[1]:
            result["eligible"] = True
            result["reason"] = f"FP3: Same hypothesis '{selected[0]}' selected twice — escalate"
            result["fp_id"] = "FP3"
            return result

    return result


def _gather_raw_evidence(project_dir, step, gate):
    """Bundle raw log content for the failing gate.

    Returns:
        dict with keys: gate_log_path (str), gate_log_excerpt (str),
                        pacs_log_excerpt (str or None).
    """
    result = {
        "gate_log_path": "",
        "gate_log_excerpt": "",
        "pacs_log_excerpt": None,
    }

    # Determine log path based on gate type
    if gate == "verification":
        log_path = os.path.join(
            project_dir, "verification-logs", f"step-{step}-verify.md"
        )
    elif gate == "pacs":
        log_path = os.path.join(
            project_dir, "pacs-logs", f"step-{step}-pacs.md"
        )
    elif gate == "review":
        log_path = os.path.join(
            project_dir, "review-logs", f"step-{step}-review.md"
        )
    else:
        return result

    result["gate_log_path"] = log_path
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                # Read only first ERROR_RESULT_CHARS to avoid OOM on large logs
                result["gate_log_excerpt"] = f.read(ERROR_RESULT_CHARS)
        except OSError:
            pass

    # Always include pacs log if available (even for non-pacs gates)
    if gate != "pacs":
        pacs_path = os.path.join(
            project_dir, "pacs-logs", f"step-{step}-pacs.md"
        )
        if os.path.exists(pacs_path):
            try:
                with open(pacs_path, "r", encoding="utf-8") as f:
                    result["pacs_log_excerpt"] = f.read(ERROR_RESULT_CHARS)
            except OSError:
                pass

    # VE cross-check: include hallucination evidence if available
    # (validate_criteria_evidence.py writes JSON to stdout, not to disk,
    #  but diagnosis can still benefit from V1d/V1e warnings in verification log)
    if gate == "verification":
        result["hallucination_check_hint"] = (
            "Run: python3 .claude/hooks/scripts/validate_criteria_evidence.py "
            f"--step {step} --project-dir {project_dir} --auto-detect"
        )

    return result


def validate_diagnosis_log(project_dir, step, gate):
    """P1 Post-validation: Verify diagnosis log structural integrity (AD1-AD10).

    Called after LLM writes the diagnosis log. All checks are deterministic.

    Args:
        project_dir: Project root path.
        step: Step number.
        gate: One of 'verification', 'pacs', 'review'.

    Returns:
        tuple(is_valid: bool, warnings: list[str])

    Checks:
        AD1: Diagnosis log file exists in diagnosis-logs/
        AD2: Minimum file size (≥ 100 bytes)
        AD3: Gate field matches expected gate
        AD4: Selected hypothesis present (H1/H2/H3/H4)
        AD5: Evidence section present (≥ 1 evidence item)
        AD6: Action plan section present
        AD7: No forward step references (source: Step N where N > step)
        AD8: Hypothesis count ≥ 2 (must consider alternatives)
        AD9: Selected hypothesis is one of the listed hypotheses
        AD10: Previous diagnosis referenced (if retry > 0)
    """
    warnings = []

    # AD1: File exists
    diag_dir = os.path.join(project_dir, "diagnosis-logs")
    # Find the latest diagnosis log for this step+gate
    diag_path = None
    if os.path.isdir(diag_dir):
        candidates = sorted([
            f for f in os.listdir(diag_dir)
            if f.startswith(f"step-{step}-{gate}-") and f.endswith(".md")
        ])
        if candidates:
            diag_path = os.path.join(diag_dir, candidates[-1])

    if not diag_path or not os.path.exists(diag_path):
        warnings.append(
            f"AD1 FAIL: No diagnosis log found for step-{step} gate={gate} "
            f"in diagnosis-logs/"
        )
        return False, warnings

    # Read content
    try:
        with open(diag_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        warnings.append(f"AD1 FAIL: Cannot read diagnosis log — {e}")
        return False, warnings

    # AD2: Minimum size
    if len(content) < 100:
        warnings.append(
            f"AD2 FAIL: Diagnosis log too small ({len(content)}B < 100B)"
        )

    # AD3: Gate field matches
    gate_match = _DIAG_GATE_RE.search(content)
    if gate_match:
        found_gate = gate_match.group(1).lower()
        if found_gate != gate.lower():
            warnings.append(
                f"AD3 FAIL: Gate mismatch — expected '{gate}', found '{found_gate}'"
            )
    else:
        warnings.append("AD3 FAIL: No Gate field found in diagnosis log")

    # AD4: Selected hypothesis present
    selected_match = _DIAG_SELECTED_RE.search(content)
    if not selected_match:
        warnings.append("AD4 FAIL: No selected hypothesis found")

    # AD5: Evidence items (≥ 1)
    evidence_items = _DIAG_EVIDENCE_RE.findall(content)
    if len(evidence_items) < 1:
        warnings.append(
            f"AD5 FAIL: Insufficient evidence items ({len(evidence_items)} < 1)"
        )

    # AD6: Action plan section
    action_plan_re = re.compile(
        r"^#+\s*(?:Action\s*Plan|Recommended\s*Action|Next\s*Steps?)\b",
        re.MULTILINE | re.IGNORECASE,
    )
    if not action_plan_re.search(content):
        warnings.append("AD6 FAIL: No Action Plan section found")

    # AD7: No forward step references
    source_refs = _DIAG_SOURCE_STEP_RE.findall(content)
    for ref_step_str in source_refs:
        ref_step = int(ref_step_str)
        if ref_step > step:
            warnings.append(
                f"AD7 FAIL: Forward reference to Step {ref_step} (current: {step})"
            )

    # AD8: Hypothesis count ≥ 2
    hypotheses_found = _DIAG_HYPOTHESIS_RE.findall(content)
    if len(hypotheses_found) < 2:
        warnings.append(
            f"AD8 FAIL: Insufficient hypotheses ({len(hypotheses_found)} < 2)"
        )

    # AD9: Selected hypothesis is one of the listed ones
    # Extract H-IDs only from hypothesis headings (not from arbitrary body text)
    if selected_match and hypotheses_found:
        listed_h_ids = set()
        for h_text in hypotheses_found:
            h_id_match = re.search(r"\bH[1-4]\b", h_text)
            if h_id_match:
                listed_h_ids.add(h_id_match.group())
        selected_h_id = re.search(
            r"\bH[1-4]\b", selected_match.group(1).strip()
        )
        if selected_h_id and selected_h_id.group() not in listed_h_ids:
            warnings.append(
                f"AD9 FAIL: Selected hypothesis '{selected_h_id.group()}' "
                f"not found among listed hypotheses {listed_h_ids}"
            )

    # AD10: Previous diagnosis referenced (if retry > 0)
    retry_history = _gather_retry_history(project_dir, step, gate)
    if retry_history["retries_used"] > 0 and retry_history["previous_diagnoses"]:
        prev_ref_re = re.compile(
            r"(?:previous|prior|earlier)\s+(?:diagnosis|attempt|retry)",
            re.IGNORECASE,
        )
        if not prev_ref_re.search(content):
            warnings.append(
                "AD10 WARNING: No reference to previous diagnosis "
                f"(retry #{retry_history['retries_used']})"
            )

    # Determine overall validity (any FAIL → invalid)
    is_valid = not any("FAIL" in w for w in warnings)
    return is_valid, warnings


def _extract_diagnosis_patterns(project_dir):
    """Extract diagnosis patterns from diagnosis-logs/ for Knowledge Archive.

    Scans diagnosis-logs/ for completed diagnosis files and extracts
    step, gate, selected_hypothesis, and evidence summary.

    Returns:
        list of dicts with keys: step, gate, selected_hypothesis, evidence_count.
    """
    patterns = []
    diag_dir = os.path.join(project_dir, "diagnosis-logs")
    if not os.path.isdir(diag_dir):
        return patterns

    try:
        for fname in sorted(os.listdir(diag_dir)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(diag_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract step and gate from filename: step-N-gate-timestamp.md
                parts = fname.replace(".md", "").split("-")
                step_num = None
                gate_name = None
                for i, p in enumerate(parts):
                    if p == "step" and i + 1 < len(parts):
                        try:
                            step_num = int(parts[i + 1])
                        except ValueError:
                            pass
                    if p in ("verification", "pacs", "review"):
                        gate_name = p

                selected = _DIAG_SELECTED_RE.search(content)
                evidence_items = _DIAG_EVIDENCE_RE.findall(content)

                patterns.append({
                    "step": step_num,
                    "gate": gate_name,
                    "selected_hypothesis": (
                        selected.group(1).strip() if selected else "unknown"
                    ),
                    "evidence_count": len(evidence_items),
                })
            except OSError:
                pass
    except OSError:
        pass

    return patterns


# ---------------------------------------------------------------------------
# Thesis state summary — shared by save_context.py & generate_context_summary.py
# ---------------------------------------------------------------------------

def get_thesis_state_summary(project_dir):
    """Read thesis SOT(s) and return a brief state summary for snapshots.

    P1 compliant: deterministic file reads only, no AI judgment.
    Non-blocking: returns empty string on any error.
    Read-only: reads session.json — never modifies it.

    Returns markdown string with step, status, gates, and HITL info.
    """
    try:
        thesis_root = os.path.join(project_dir, "thesis-output")
        if not os.path.isdir(thesis_root):
            return ""

        summaries = []
        for proj_name in sorted(os.listdir(thesis_root)):
            sot_path = os.path.join(thesis_root, proj_name, "session.json")
            if not os.path.isfile(sot_path):
                continue
            with open(sot_path, "r", encoding="utf-8") as f:
                sot = json.load(f)

            step = sot.get("current_step", 0)
            total = sot.get("total_steps", "?")
            status = sot.get("status", "unknown")
            rtype = sot.get("research_type", "undecided")

            # Gate summary
            gates = sot.get("gates", {})
            gate_str = ", ".join(
                f"{k}:{v.get('status', v) if isinstance(v, dict) else v}"
                for k, v in gates.items()
            ) if gates else "none"

            # HITL summary
            hitl = sot.get("hitl_checkpoints", {})
            completed_hitl = [
                k for k, v in hitl.items()
                if (v.get("status") if isinstance(v, dict) else v) == "completed"
            ]
            hitl_str = ", ".join(completed_hitl) if completed_hitl else "none"

            summaries.append(
                f"  - **{proj_name}**: step {step}/{total}, "
                f"status={status}, type={rtype}\n"
                f"    - Gates: {gate_str}\n"
                f"    - HITL completed: {hitl_str}"
            )

        if not summaries:
            return ""

        return (
            "\n\n## Thesis Workflow State\n\n"
            + "\n".join(summaries)
            + "\n"
        )
    except Exception:
        return ""
