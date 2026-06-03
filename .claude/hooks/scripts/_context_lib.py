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
from _snapshot_lib import *  # noqa: F401,F403
from _snapshot_lib import (  # noqa: F401 — underscore not covered by import *
    _extract_decisions,
    _remove_section,
)
from _capture_lib import *  # noqa: F401,F403
# Underscore-prefixed symbols imported by external scripts are not covered
# by `import *`, so re-export them explicitly:
from _validation_lib import (  # noqa: F401
    _DKS_REF_RE,
    _find_translation_files_for_step,
    _TRACE_MARKER_RE,
)
from _core_lib import (  # noqa: F401 — underscore symbols not covered by import *
    _truncate,
    _DIAG_GATE_RE,
    _DIAG_SELECTED_RE,
)
# Claude's context window (200K tokens)
CONTEXT_WINDOW_TOKENS = 200_000
# System prompt overhead estimate (tokens)
SYSTEM_OVERHEAD_TOKENS = 15_000
# Effective capacity
EFFECTIVE_CAPACITY = CONTEXT_WINDOW_TOKENS - SYSTEM_OVERHEAD_TOKENS
# 75% threshold
THRESHOLD_75_TOKENS = int(EFFECTIVE_CAPACITY * 0.75)
# Knowledge Archive limits (Area 1: Cross-Session Knowledge Archive)
MAX_KNOWLEDGE_INDEX_ENTRIES = 200
MAX_SESSION_ARCHIVES = 20

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

# Used by Abductive Diagnosis functions — diagnosis-logs/ parsing
# Captures the FULL heading line (including H-ID) so AD9 can extract H[1-4] from it.
# E.g., "## H1: Upstream data quality issue" → captures "H1: Upstream data quality issue"
_DIAG_HYPOTHESIS_RE = re.compile(
    r"^#+\s*((?:H\d|Hypothesis)\b.+)", re.MULTILINE | re.IGNORECASE,
)
_DIAG_EVIDENCE_RE = re.compile(
    r"^-\s*\*?\*?Evidence\*?\*?\s*:\s*(.+)", re.MULTILINE | re.IGNORECASE,
)
_DIAG_SOURCE_STEP_RE = re.compile(
    r"\(source:\s*Step\s+(\d+)\)", re.IGNORECASE,
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


def read_stdin_json():
    """Read and parse JSON from stdin (hook input)."""
    try:
        raw = sys.stdin.read()
        if raw.strip():
            return json.loads(raw)
    except (json.JSONDecodeError, Exception):
        pass
    return {}


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
