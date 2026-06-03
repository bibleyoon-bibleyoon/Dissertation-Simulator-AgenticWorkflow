#!/usr/bin/env python3
"""Snapshot generation, compression, and decision/quality-gate extraction.

Extracted from _context_lib.py per ADR-078 (Increment 3). Builds context
snapshots, compresses them under budget, and manages snapshot files.
Depends on _core_lib (foundation) and _capture_lib (transcript/SOT/git capture).
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

# Ensure sibling modules resolve under file-path loading (ADR-076/077/078).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _core_lib import (
    CHARS_PER_TOKEN,
    _DIAG_GATE_RE,
    _DIAG_SELECTED_RE,
    _truncate,
    atomic_write,
    validate_sot_schema,
)
from _capture_lib import (
    capture_git_state,
    check_ulw_compliance,
    detect_conversation_phase,
    detect_phase_transitions,
    detect_ulw_mode,
    extract_completion_state,
    read_active_team_state,
    read_autopilot_state,
)
from _validation_lib import (  # quality-gate/anti-skip validators used by snapshot
    parse_review_verdict,
    validate_step_output,
)


MAX_SNAPSHOT_CHARS = 100_000


DEDUP_WINDOW_SECONDS = 5


STOP_DEDUP_WINDOW_SECONDS = 30


MAX_SNAPSHOTS = {
    "precompact": 3,
    "sessionend": 3,
    "threshold": 2,
    "stop": 5,
}


DEFAULT_MAX_SNAPSHOTS = 3


E5_RICH_CONTENT_MARKER = "### 수정 중이던 파일"


E5_COMPLETION_STATE_MARKER = "## 결정론적 완료 상태"


E5_DESIGN_DECISIONS_MARKER = "## 주요 설계 결정"


E5_RICH_SIGNALS = [
    E5_RICH_CONTENT_MARKER,         # "### 수정 중이던 파일"
    E5_COMPLETION_STATE_MARKER,     # "## 결정론적 완료 상태"
    E5_DESIGN_DECISIONS_MARKER,     # "## 주요 설계 결정"
]


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


_NEXT_STEP_RE = re.compile(
    r'(?:다음으로|이제|그 다음|그 후|Next,?|Now |Then )'
    r'\s*(.{10,500}?)(?:\.\s|\n\n|$)',
    re.MULTILINE,
)


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


_SYSTEM_CMD_RE = re.compile(
    r'^\s*<command-name>|^\s*/(?:clear|help|compact|init|resume|review|login|logout|mcp|config)\b',
    re.IGNORECASE | re.MULTILINE,
)


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
