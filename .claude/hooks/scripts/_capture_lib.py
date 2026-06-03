#!/usr/bin/env python3
"""Session capture primitives extracted from _context_lib.py per ADR-077 (Increment 2).

Transcript parsing, SOT/git/completion-state capture, ULW detection, and
conversation-phase classification. Depends only on _core_lib.
"""

import json
import os
import re
import sys
import subprocess
from datetime import datetime

# Ensure _core_lib resolves under file-path loading (ADR-076/077).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _core_lib import (
    BASH_CMD_CHARS,
    EDIT_PREVIEW_CHARS,
    ERROR_RESULT_CHARS,
    GENERIC_INPUT_CHARS,
    NORMAL_RESULT_CHARS,
    SOT_CAPTURE_CHARS,
    TASK_PROMPT_CHARS,
    TOOL_ERROR_PATTERNS,
    WRITE_PREVIEW_CHARS,
    _truncate,
    sot_paths,
)


def parse_transcript(transcript_path):
    """
    Parse a Claude Code transcript JSONL file into structured entries.

    Returns list of dicts with keys:
        - type: 'user_message', 'assistant_text', 'tool_use', 'tool_result'
        - timestamp: ISO string
        - content: extracted content (varies by type)
        - file_path: (tool_use only, Write/Edit) deterministic file path
        - line_count: (tool_use only, Write) number of lines
    """
    entries = []
    if not transcript_path or not os.path.exists(transcript_path):
        return entries

    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = obj.get("type")
                timestamp = obj.get("timestamp", "")

                if entry_type == "user":
                    entries.extend(_parse_user_entry(obj, timestamp))
                elif entry_type == "assistant":
                    entries.extend(_parse_assistant_entry(obj, timestamp))
                # Skip: progress, file-history-snapshot, system
    except Exception:
        pass

    return entries


def _parse_user_entry(obj, timestamp):
    """Extract user messages and tool results from user-type entries."""
    results = []
    message = obj.get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        # Plain text user message
        text = content.strip()
        if text and not text.startswith("<local-command-"):
            results.append({
                "type": "user_message",
                "timestamp": timestamp,
                "content": text,
            })
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")

            if block_type == "text":
                text = block.get("text", "").strip()
                if text and not text.startswith("<local-command-"):
                    results.append({
                        "type": "user_message",
                        "timestamp": timestamp,
                        "content": text,
                    })

            elif block_type == "tool_result":
                tool_content = block.get("content", "")
                is_error = block.get("is_error", False)
                summary = _extract_tool_result_summary(tool_content)
                if summary:
                    results.append({
                        "type": "tool_result",
                        "timestamp": timestamp,
                        "tool_use_id": block.get("tool_use_id", ""),
                        "is_error": is_error,
                        "content": summary,
                    })

    return results


def _parse_assistant_entry(obj, timestamp):
    """Extract assistant text and tool uses from assistant-type entries.

    For tool_use entries, structured metadata (file_path, line_count) is
    extracted directly from tool_input — NOT parsed from summary strings.
    This ensures 100% deterministic, accurate file operation tracking.
    """
    results = []
    message = obj.get("message", {})
    content = message.get("content", [])

    if isinstance(content, str):
        text = content.strip()
        if text:
            results.append({
                "type": "assistant_text",
                "timestamp": timestamp,
                "content": _truncate(text, 5000),
            })
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")

            if block_type == "text":
                text = block.get("text", "").strip()
                if text:
                    results.append({
                        "type": "assistant_text",
                        "timestamp": timestamp,
                        "content": _truncate(text, 5000),
                    })

            elif block_type == "tool_use":
                tool_name = block.get("name", "unknown")
                tool_input = block.get("input", {})
                summary = _extract_tool_use_summary(tool_name, tool_input)

                entry = {
                    "type": "tool_use",
                    "timestamp": timestamp,
                    "tool_name": tool_name,
                    "tool_use_id": block.get("id", ""),
                    "content": summary,
                }

                # Structured metadata — deterministic, no string parsing
                if tool_name == "Write":
                    entry["file_path"] = tool_input.get("file_path", "")
                    file_content = tool_input.get("content", "")
                    entry["line_count"] = len(file_content.split("\n")) if file_content else 0
                elif tool_name == "Edit":
                    entry["file_path"] = tool_input.get("file_path", "")
                elif tool_name == "Bash":
                    entry["command"] = tool_input.get("command", "")
                    entry["description"] = tool_input.get("description", "")
                elif tool_name == "Read":
                    entry["file_path"] = tool_input.get("file_path", "")

                results.append(entry)

    return results


def _extract_tool_use_summary(tool_name, tool_input):
    """Apply per-tool extraction rules to keep snapshots compact."""
    if tool_name in ("Write",):
        path = tool_input.get("file_path", "unknown")
        content = tool_input.get("content", "")
        lines = content.split("\n")
        preview = "\n".join(lines[:3])
        return f"Write → {path} ({len(lines)} lines)\n  Preview: {_truncate(preview, WRITE_PREVIEW_CHARS)}"

    elif tool_name in ("Edit",):
        path = tool_input.get("file_path", "unknown")
        old = tool_input.get("old_string", "")
        new = tool_input.get("new_string", "")
        # B-1: 첫 5줄 × EDIT_PREVIEW_CHARS — "왜" 그 편집을 했는지 의도+맥락 보존
        old_preview = "\n".join(old.split("\n")[:5]) if old else ""
        new_preview = "\n".join(new.split("\n")[:5]) if new else ""
        return (f"Edit → {path}\n"
                f"  OLD: {_truncate(old_preview, EDIT_PREVIEW_CHARS)}\n"
                f"  NEW: {_truncate(new_preview, EDIT_PREVIEW_CHARS)}")

    elif tool_name in ("Read",):
        path = tool_input.get("file_path", "unknown")
        return f"Read → {path}"

    elif tool_name in ("Bash",):
        cmd = tool_input.get("command", "")
        desc = tool_input.get("description", "")
        return f"Bash: {_truncate(cmd, BASH_CMD_CHARS)}" + (f" ({desc})" if desc else "")

    elif tool_name in ("Task",):
        desc = tool_input.get("description", "")
        prompt = tool_input.get("prompt", "")
        agent_type = tool_input.get("subagent_type", "")
        return f"Task ({agent_type}): {desc}\n  Prompt: {_truncate(prompt, TASK_PROMPT_CHARS)}"

    elif tool_name in ("Glob",):
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", "")
        return f"Glob: {pattern}" + (f" in {path}" if path else "")

    elif tool_name in ("Grep",):
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", "")
        return f"Grep: {pattern}" + (f" in {path}" if path else "")

    elif tool_name in ("WebSearch",):
        query = tool_input.get("query", "")
        return f"WebSearch: {query}"

    elif tool_name in ("WebFetch",):
        url = tool_input.get("url", "")
        return f"WebFetch: {_truncate(url, 100)}"

    else:
        # Generic: show first GENERIC_INPUT_CHARS of input
        return f"{tool_name}: {_truncate(json.dumps(tool_input, ensure_ascii=False), GENERIC_INPUT_CHARS)}"


def _extract_tool_result_summary(content):
    """Extract summary from tool_result content.

    C-3: Error recovery narrative — error-containing results get expanded
    truncation limit (ERROR_RESULT_CHARS) to preserve diagnostic context.
    """
    _ERROR_PATTERNS = ("error", "Error", "ERROR", "failed", "Failed", "FAILED",
                       "traceback", "Traceback", "exception", "Exception")

    def _limit_for(text):
        if any(pat in text for pat in _ERROR_PATTERNS):
            return ERROR_RESULT_CHARS  # B-2: 에러 메시지 전체 보존 (stack trace 포함)
        return NORMAL_RESULT_CHARS

    if isinstance(content, str):
        return _truncate(content, _limit_for(content))
    elif isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        combined = "\n".join(texts)
        return _truncate(combined, _limit_for(combined))
    return ""


def capture_sot(project_dir):
    """
    Read SOT file (state.yaml) if it exists.
    Hook is READ-ONLY for SOT — only captures content.
    """
    for sot_path in sot_paths(project_dir):
        if os.path.exists(sot_path):
            try:
                with open(sot_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return {
                    "path": sot_path,
                    "content": _truncate(content, SOT_CAPTURE_CHARS),
                    "mtime": datetime.fromtimestamp(
                        os.path.getmtime(sot_path)
                    ).isoformat(),
                }
            except Exception:
                pass

    return None


def read_autopilot_state(project_dir):
    """Read autopilot state from SOT (state.yaml). Read-only.

    Returns dict with autopilot fields if enabled, None otherwise.

    IMPORTANT: Does NOT use capture_sot() — reads state.yaml directly
    without truncation. capture_sot() truncates to 3000 chars (for snapshot
    display), which can cut the autopilot section in large SOT files.

    Schema compatibility: Supports both AGENTS.md schema (workflow.autopilot)
    and flat schema (top-level autopilot). AGENTS.md §5.1 is authoritative.

    P1 Compliance: All fields are deterministic extractions from YAML/regex.
    SOT Compliance: Read-only file access.
    """
    # Direct file read — uses sot_paths() for consistency (A-3)
    # Only YAML files (not JSON) — autopilot regex patterns assume YAML format
    # CQ-1: Renamed to avoid shadowing the sot_paths() function
    yaml_sot_paths = [p for p in sot_paths(project_dir) if not p.endswith(".json")]

    content = ""
    for sot_path in yaml_sot_paths:
        if os.path.exists(sot_path):
            try:
                with open(sot_path, "r", encoding="utf-8") as f:
                    content = f.read()
                break
            except Exception:
                continue

    if not content:
        return None

    # Try PyYAML first (precise structured parsing)
    try:
        import yaml
        data = yaml.safe_load(content)
        if isinstance(data, dict):
            # Schema compatibility: check both locations
            # AGENTS.md §5.1 schema: workflow.autopilot.enabled
            # Flat schema: autopilot.enabled (top-level)
            wf = data.get("workflow", {})
            if not isinstance(wf, dict):
                wf = {}
            ap = wf.get("autopilot") or data.get("autopilot")
            if not isinstance(ap, dict) or not ap.get("enabled"):
                return None
            return {
                "enabled": True,
                "activated_at": ap.get("activated_at", ""),
                "auto_approved_steps": ap.get("auto_approved_steps", []),
                "current_step": wf.get("current_step", 0),
                "workflow_name": wf.get("name", ""),
                "workflow_status": wf.get("status", ""),
                "outputs": wf.get("outputs", {}),
            }
    except Exception:
        pass

    # Regex fallback (when PyYAML is not available)
    # Matches both "autopilot:\n  enabled: true" at any nesting level
    enabled_match = re.search(
        r'autopilot\s*:\s*\n\s+enabled\s*:\s*(true|yes)',
        content, re.IGNORECASE
    )
    if not enabled_match:
        return None

    state = {
        "enabled": True,
        "activated_at": "",
        "auto_approved_steps": [],
        "current_step": 0,
        "workflow_name": "",
        "workflow_status": "",
        "outputs": {},
    }

    for field, pattern in [
        ("activated_at", r'activated_at\s*:\s*["\']?(.+?)["\']?\s*$'),
        ("current_step", r'current_step\s*:\s*(\d+)'),
        ("workflow_name", r'name\s*:\s*["\']?(.+?)["\']?\s*$'),
        ("workflow_status", r'status\s*:\s*["\']?(.+?)["\']?\s*$'),
    ]:
        m = re.search(pattern, content, re.MULTILINE)
        if m:
            val = m.group(1).strip()
            state[field] = int(val) if field == "current_step" else val

    # Extract auto_approved_steps list
    steps_match = re.search(r'auto_approved_steps\s*:\s*\[([^\]]*)\]', content)
    if steps_match:
        steps_str = steps_match.group(1)
        state["auto_approved_steps"] = [
            int(s.strip()) for s in steps_str.split(",")
            if s.strip().isdigit()
        ]

    # Extract outputs map
    outputs_section = re.search(
        r'outputs\s*:\s*\n((?:\s+step-\d+\s*:.+\n?)*)', content
    )
    if outputs_section:
        for m in re.finditer(
            r'(step-\d+)\s*:\s*["\']?(.+?)["\']?\s*$',
            outputs_section.group(1), re.MULTILINE
        ):
            state["outputs"][m.group(1)] = m.group(2).strip()

    return state


def read_active_team_state(project_dir):
    """Read active_team state from SOT (state.yaml). Read-only.

    Returns dict with active_team fields if a team is active, None otherwise.
    This enables 2-Layer RLM: Layer 1 (auto snapshots) + Layer 2 (team summaries in SOT).

    Schema (from claude-code-patterns.md §SOT 갱신 프로토콜):
      active_team:
        name: "team-name"
        status: "partial" | "all_completed"
        tasks_completed: ["task-1", ...]
        tasks_pending: ["task-2", ...]
        completed_summaries:
          task-1:
            agent: "@researcher"
            model: "sonnet"
            output: "path/to/output.md"
            summary: "brief description"

    P1 Compliance: All fields are deterministic extractions from YAML/regex.
    SOT Compliance: Read-only file access.
    """
    # A-3: use sot_paths() — YAML only (regex parsing)
    # B-1: Renamed to avoid shadowing the sot_paths() function (same fix as CQ-1)
    yaml_sot_paths = [p for p in sot_paths(project_dir) if not p.endswith(".json")]

    content = ""
    for sot_path in yaml_sot_paths:
        if os.path.exists(sot_path):
            try:
                with open(sot_path, "r", encoding="utf-8") as f:
                    content = f.read()
                break
            except Exception:
                continue

    if not content:
        return None

    # Try PyYAML first (precise structured parsing)
    try:
        import yaml
        data = yaml.safe_load(content)
        if isinstance(data, dict):
            # Check both nested (workflow.active_team) and flat (active_team)
            wf = data.get("workflow", {})
            if not isinstance(wf, dict):
                wf = {}
            at = wf.get("active_team") or data.get("active_team")
            if not isinstance(at, dict) or not at.get("name"):
                return None
            return {
                "name": at.get("name", ""),
                "status": at.get("status", "unknown"),
                "tasks_completed": at.get("tasks_completed", []),
                "tasks_pending": at.get("tasks_pending", []),
                "completed_summaries": at.get("completed_summaries", {}),
            }
    except Exception:
        pass

    # Regex fallback (when PyYAML is not available)
    name_match = re.search(
        r'active_team\s*:\s*\n\s+name\s*:\s*["\']?(.+?)["\']?\s*$',
        content, re.MULTILINE
    )
    if not name_match:
        return None

    state = {
        "name": name_match.group(1).strip(),
        "status": "unknown",
        "tasks_completed": [],
        "tasks_pending": [],
        "completed_summaries": {},
    }

    status_match = re.search(
        r'active_team\s*:.*?status\s*:\s*["\']?(\w+)["\']?',
        content, re.DOTALL
    )
    if status_match:
        state["status"] = status_match.group(1).strip()

    # Extract task lists (YAML inline array format)
    for field in ["tasks_completed", "tasks_pending"]:
        m = re.search(rf'{field}\s*:\s*\[([^\]]*)\]', content)
        if m:
            items = [s.strip().strip("\"'") for s in m.group(1).split(",") if s.strip()]
            state[field] = items

    return state


def detect_ulw_mode(entries):
    """Detect ULW (Ultrawork) mode activation from user messages.

    Scans transcript entries for the "ulw" keyword in user messages.
    Uses word-boundary regex to prevent false positives from variable names,
    file paths, or URLs (e.g., "resultw", "/usr/local/ulwrap").

    Args:
        entries: List of parsed transcript entries.

    Returns:
        dict with {active, detected_in, source_message, message_index} or None.

    P1 Compliance: Deterministic regex match on verbatim user messages.
    """
    # Word-boundary pattern: not preceded/followed by alphanumeric, underscore, slash, dot, hyphen
    ULW_PATTERN = re.compile(
        r'(?<![a-zA-Z0-9_/\-\.])ulw(?![a-zA-Z0-9_/\-\.])',
        re.IGNORECASE,
    )

    user_messages = [
        (i, e) for i, e in enumerate(entries)
        if e.get("type") == "user_message"
        and not (e.get("content", "").startswith("<") and ">" in e.get("content", "")[:50])
    ]

    for idx, (msg_index, entry) in enumerate(user_messages):
        content = entry.get("content", "")
        if ULW_PATTERN.search(content):
            return {
                "active": True,
                "detected_in": "first" if idx == 0 else "subsequent",
                "source_message": content[:500],
                "message_index": msg_index,
            }

    return None


def _extract_file_from_nearby_tool_use(entries, error_idx, window=3):
    """Extract file path from tool_use entries near an error (private helper).

    Looks backward from error_idx within `window` entries for Edit/Write
    tool_use with a file_path field.

    Note: parse_transcript() stores file_path at the entry's top level
    (not nested under "parameters" or "input") — see line 341/345 of
    _parse_assistant_entry().

    Returns:
        str or None: file path if found.
    """
    start = max(0, error_idx - window)
    for i in range(error_idx - 1, start - 1, -1):
        entry = entries[i]
        if entry.get("type") == "tool_use":
            name = entry.get("tool_name", "")
            if name in ("Edit", "Write"):
                # file_path is a top-level key set by parse_transcript()
                fp = entry.get("file_path", "")
                if fp:
                    return fp
    return None


def check_ulw_compliance(entries):
    """ULW 모드 활성 시 3개 강화 규칙(Intensifiers)의 준수 여부를 결정론적으로 검증.

    All checks are pure counting and pattern matching — P1 compliant.
    No heuristic inference. No AI judgment.

    Intensifiers:
      I-1. Sisyphus Persistence: error recovery + no partial completion (max 3 retries)
      I-2. Mandatory Task Decomposition: TaskCreate/TaskUpdate/TaskList usage
      I-3. Bounded Retry Escalation: no more than 3 consecutive retries on same target

    Args:
        entries: List of parsed transcript entries.

    Returns:
        dict with compliance metrics and warnings, or None if ULW inactive.
    """
    ulw_state = detect_ulw_mode(entries)
    if not ulw_state:
        return None

    # Filter: only count entries AFTER ULW activation point
    # Prevents false positives when ULW is activated in a "subsequent" message
    ulw_start_idx = ulw_state["message_index"]
    post_ulw_entries = entries[ulw_start_idx:]

    tool_uses = [e for e in post_ulw_entries if e.get("type") == "tool_use"]

    compliance = {
        "active": True,
        "task_creates": 0,
        "task_updates": 0,
        "task_lists": 0,
        "total_tool_uses": len(tool_uses),
        "errors_detected": 0,
        "post_error_actions": 0,
        "max_consecutive_retries": 0,
        "warnings": [],
    }

    # Count task management tool uses
    for tu in tool_uses:
        name = tu.get("tool_name", "")
        if name == "TaskCreate":
            compliance["task_creates"] += 1
        elif name == "TaskUpdate":
            compliance["task_updates"] += 1
        elif name == "TaskList":
            compliance["task_lists"] += 1

    # Detect errors and post-error recovery attempts
    # Uses module-level TOOL_ERROR_PATTERNS (DRY — shared with extract_completion_state)
    last_error_global_idx = -1
    # Track consecutive retries on same file for I-3
    error_file_sequence = []  # list of (file_path_or_None,)

    for i, entry in enumerate(post_ulw_entries):
        if entry.get("type") == "tool_result":
            is_error = entry.get("is_error", False)
            content = entry.get("content", "")[:500]
            if is_error or any(sig in content for sig in TOOL_ERROR_PATTERNS):
                compliance["errors_detected"] += 1
                last_error_global_idx = i
                fp = _extract_file_from_nearby_tool_use(post_ulw_entries, i)
                error_file_sequence.append(fp)

    # Count tool uses that occurred AFTER the last error (recovery attempts)
    if last_error_global_idx >= 0:
        for i, entry in enumerate(post_ulw_entries):
            if i > last_error_global_idx and entry.get("type") == "tool_use":
                compliance["post_error_actions"] += 1

    # I-3: Detect max consecutive retries on same file
    if error_file_sequence:
        max_consecutive = 1
        current_run = 1
        for j in range(1, len(error_file_sequence)):
            prev_fp = error_file_sequence[j - 1]
            curr_fp = error_file_sequence[j]
            if prev_fp and curr_fp and prev_fp == curr_fp:
                current_run += 1
                if current_run > max_consecutive:
                    max_consecutive = current_run
            else:
                current_run = 1
        compliance["max_consecutive_retries"] = max_consecutive

    # Generate deterministic warnings — mapped to 3 Intensifiers

    # W1 (I-1 Sisyphus Persistence): Errors detected but no subsequent actions
    if compliance["errors_detected"] > 0 and compliance["post_error_actions"] == 0:
        compliance["warnings"].append(
            "ULW_NO_SISYPHUS: 에러 {}건 감지, 후속 조치 0건 — I-1 Sisyphus Persistence 미준수".format(
                compliance["errors_detected"]
            )
        )

    # W2 (I-2 Mandatory Task Decomposition): No task tracking despite significant tool usage
    if compliance["task_creates"] == 0 and compliance["total_tool_uses"] >= 5:
        compliance["warnings"].append(
            "ULW_NO_DECOMPOSITION: 도구 {}회 사용, TaskCreate 0회 — I-2 Mandatory Task Decomposition 미준수".format(
                compliance["total_tool_uses"]
            )
        )

    # W2a (I-2 sub): Tasks created but never updated (no progress tracking)
    if compliance["task_creates"] > 0 and compliance["task_updates"] == 0:
        compliance["warnings"].append(
            "ULW_NO_PROGRESS: TaskCreate {}회, TaskUpdate 0회 — I-2 Progress Tracking 미준수".format(
                compliance["task_creates"]
            )
        )

    # W2b (I-2 sub): Tasks created but never listed (no completion verification)
    if compliance["task_creates"] > 0 and compliance["task_lists"] == 0:
        compliance["warnings"].append(
            "ULW_NO_VERIFY: TaskCreate {}회, TaskList 0회 — I-2 완료 검증 미수행".format(
                compliance["task_creates"]
            )
        )

    # W3 (I-3 Bounded Retry Escalation): Same target retried > 3 times consecutively
    if compliance["max_consecutive_retries"] > 3:
        compliance["warnings"].append(
            "ULW_RETRY_EXCEEDED: 동일 대상 연속 재시도 {}회 — I-3 Bounded Retry 초과 (최대 3회)".format(
                compliance["max_consecutive_retries"]
            )
        )

    return compliance


def capture_git_state(project_dir, max_diff_chars=8000):
    """Git 변경 상태의 결정론적 캡처 (읽기 전용, SOT 준수).

    3개 시그널을 캡처하여 모든 시나리오에서 ground-truth 제공:
    1. git status --porcelain  (현재 작업 트리 상태)
    2. git diff HEAD            (커밋되지 않은 변경)
    3. git log --oneline --stat -5  (최근 커밋 — post-commit 시나리오 대응)

    P1 Compliance: All fields are subprocess stdout captures (deterministic).
    SOT Compliance: git commands are read-only.
    """
    result = {"status": "", "diff_stat": "", "diff_content": "", "recent_commits": ""}

    def _run_git(args, max_chars=2000):
        try:
            proc = subprocess.run(
                ["git"] + args,
                cwd=project_dir, capture_output=True, text=True, timeout=5
            )
            return proc.stdout.strip()[:max_chars] if proc.returncode == 0 else ""
        except Exception:
            return ""

    result["status"] = _run_git(["status", "--porcelain"])
    result["diff_stat"] = _run_git(["diff", "--stat", "HEAD"])
    result["diff_content"] = _run_git(["diff", "HEAD"], max_chars=max_diff_chars)
    result["recent_commits"] = _run_git(
        ["log", "--oneline", "--stat", "-5"], max_chars=3000
    )

    return result


def extract_completion_state(entries, project_dir):
    """결정론적 완료 상태 추출 — Claude 해석 불필요.

    P1 Compliance: All fields are deterministic extractions from
    transcript entries + filesystem checks. Zero heuristic inference.

    Hallucination prevention: Claude reads FACTS, not guesses.
    - Tool call success/failure via tool_use_id ↔ tool_result matching
    - File existence via os.path.exists() at save time
    - Quantitative metrics via counting
    """
    tool_uses = [e for e in entries if e["type"] == "tool_use"]
    tool_results = [e for e in entries if e["type"] == "tool_result"]

    # 1. Tool call counts (deterministic aggregation)
    tool_counts = {}
    for tu in tool_uses:
        name = tu.get("tool_name", "unknown")
        tool_counts[name] = tool_counts.get(name, 0) + 1

    # 2. Build tool_result lookup by tool_use_id
    result_by_id = {}
    for tr in tool_results:
        tid = tr.get("tool_use_id", "")
        if not tid:
            continue
        content = tr.get("content", "")
        is_error = tr.get("is_error", False)
        # Supplementary error pattern matching (defensive — in case is_error is missing)
        # Uses module-level TOOL_ERROR_PATTERNS (DRY — shared with check_ulw_compliance)
        has_error_pattern = any(p in content for p in TOOL_ERROR_PATTERNS) if not is_error else False
        result_by_id[tid] = is_error or has_error_pattern

    # 3. Edit/Write success/failure counts (matched via tool_use_id)
    edit_success = 0
    edit_fail = 0
    write_success = 0
    write_fail = 0
    bash_success = 0
    bash_fail = 0

    for tu in tool_uses:
        tid = tu.get("tool_use_id", "")
        name = tu.get("tool_name", "")
        is_err = result_by_id.get(tid, False)

        if name == "Edit":
            if is_err:
                edit_fail += 1
            else:
                edit_success += 1
        elif name == "Write":
            if is_err:
                write_fail += 1
            else:
                write_success += 1
        elif name == "Bash":
            if is_err:
                bash_fail += 1
            else:
                bash_success += 1

    # 4. File existence verification (filesystem check at save time)
    file_verification = []
    modified_paths = []
    seen_paths = set()
    for tu in tool_uses:
        if tu.get("tool_name") in ("Edit", "Write"):
            path = tu.get("file_path", "")
            if path and path not in seen_paths:
                seen_paths.add(path)
                modified_paths.append(path)
                exists = os.path.exists(path)
                mtime = ""
                if exists:
                    try:
                        mtime = datetime.fromtimestamp(
                            os.path.getmtime(path)
                        ).strftime("%H:%M:%S")
                    except Exception:
                        pass
                file_verification.append({
                    "path": path,
                    "exists": exists,
                    "mtime": mtime,
                })

    # 5. Session timeline (deterministic timestamps)
    timestamps = [e.get("timestamp", "") for e in entries if e.get("timestamp")]
    first_ts = timestamps[0] if timestamps else ""
    last_ts = timestamps[-1] if timestamps else ""

    return {
        "tool_counts": tool_counts,
        "edit_success": edit_success,
        "edit_fail": edit_fail,
        "write_success": write_success,
        "write_fail": write_fail,
        "bash_success": bash_success,
        "bash_fail": bash_fail,
        "file_verification": file_verification,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "total_tool_calls": len(tool_uses),
        "total_results": len(tool_results),
    }


def _classify_phase(tool_uses):
    """Classify a set of tool uses into a single phase.

    P1 Compliance: Deterministic classification based on tool proportions.
    Returns: 'research', 'planning', 'implementation', 'orchestration', or 'unknown'
    """
    if not tool_uses:
        return "unknown"

    read_tools = sum(1 for t in tool_uses if t.get("tool_name") in
                     ("Read", "Grep", "Glob", "WebSearch", "WebFetch"))
    write_tools = sum(1 for t in tool_uses if t.get("tool_name") in
                      ("Edit", "Write", "Bash"))
    plan_tools = sum(1 for t in tool_uses if t.get("tool_name") in
                     ("AskUserQuestion", "EnterPlanMode", "ExitPlanMode"))
    task_tools = sum(1 for t in tool_uses if t.get("tool_name") in
                     ("Task", "TaskCreate", "TaskUpdate", "TeamCreate", "SendMessage"))

    total = len(tool_uses)

    if plan_tools > 0 and plan_tools >= write_tools:
        return "planning"
    if task_tools > total * 0.3:
        return "orchestration"
    if read_tools > total * 0.6:
        return "research"
    if write_tools > total * 0.4:
        return "implementation"
    if read_tools > write_tools:
        return "research"
    return "implementation"


def detect_conversation_phase(tool_uses):
    """Detect current conversation phase from tool usage patterns.

    P1 Compliance: Deterministic classification based on tool proportions.
    Returns: 'research', 'planning', 'implementation', 'orchestration', or 'unknown'
    """
    return _classify_phase(tool_uses)


def detect_phase_transitions(tool_uses, window_size=20):
    """B-4: Detect phase transitions within a session.

    Splits tool_uses into sliding windows and classifies each,
    identifying where the phase changed (e.g., research → implementation).

    P1 Compliance: Deterministic — window-based classification.
    Returns: list of (phase, start_index, end_index) tuples.
    """
    if not tool_uses or len(tool_uses) < window_size:
        return [(_classify_phase(tool_uses), 0, len(tool_uses))]

    phases = []
    current_phase = None
    phase_start = 0

    for i in range(0, len(tool_uses), window_size // 2):  # 50% overlap
        window = tool_uses[i:i + window_size]
        phase = _classify_phase(window)

        if phase != current_phase:
            if current_phase is not None:
                phases.append((current_phase, phase_start, i))
            current_phase = phase
            phase_start = i

    # Add final phase
    if current_phase is not None:
        phases.append((current_phase, phase_start, len(tool_uses)))

    return phases if phases else [("unknown", 0, len(tool_uses))]
