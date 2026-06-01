#!/usr/bin/env python3
"""
Parse Claude Code JSONL session files into readable conversation transcripts.

Extracts human↔Claude text exchanges, summarizes tool calls, and saves
as clean markdown files for future reference and session loading.

Usage:
    # Parse current session (auto-detects latest JSONL):
    python scripts/parse_session_conversation.py

    # Parse specific session:
    python scripts/parse_session_conversation.py --file path/to/session.jsonl

    # Parse all sessions:
    python scripts/parse_session_conversation.py --all

    # Auto-save mode (for cron — parse latest, skip if already parsed):
    python scripts/parse_session_conversation.py --auto

Output: titan-docs/conversations/CONVERSATION_{date}_{session_id_short}.md
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Directories
JSONL_DIR = os.path.expanduser(
    "~/.claude/projects/-home-antigravity-projects-titan")
OUTPUT_DIR = "titan-docs/conversations"

# Tool call summarization — show what was done, not the full output
TOOL_SUMMARY_MAX = 120  # Max chars for tool call summary


def parse_jsonl_to_conversation(jsonl_path: str) -> list[dict]:
    """Parse JSONL file into ordered conversation entries.

    Returns list of:
        {"role": "human"|"claude"|"tool_use"|"tool_result",
         "content": str, "timestamp": str, "metadata": dict}
    """
    entries = []

    with open(jsonl_path) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            entry_type = d.get("type", "")
            timestamp = d.get("timestamp", "")
            msg = d.get("message", {})

            if entry_type == "user" and isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Multi-part content (text + images)
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    content = "\n".join(text_parts)

                if content and content.strip():
                    # Filter out system-reminder tags from user content
                    clean = _strip_system_tags(content)
                    if clean.strip():
                        entries.append({
                            "role": "human",
                            "content": clean.strip(),
                            "timestamp": timestamp,
                            "metadata": {},
                        })

            elif entry_type == "assistant" and isinstance(msg, dict):
                content_parts = msg.get("content", [])
                if isinstance(content_parts, str):
                    content_parts = [{"type": "text", "text": content_parts}]

                text_parts = []
                tool_calls = []

                for item in content_parts:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        text = item.get("text", "").strip()
                        if text:
                            text_parts.append(text)
                    elif item.get("type") == "tool_use":
                        tool_name = item.get("name", "?")
                        tool_input = item.get("input", {})
                        summary = _summarize_tool_call(tool_name, tool_input)
                        tool_calls.append(summary)

                # Combine text and tool summaries
                full_content = "\n\n".join(text_parts)

                if full_content or tool_calls:
                    entries.append({
                        "role": "claude",
                        "content": full_content,
                        "timestamp": timestamp,
                        "metadata": {"tool_calls": tool_calls},
                    })

    return entries


def _strip_system_tags(text: str) -> str:
    """Remove <system-reminder> tags and their content."""
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text,
                  flags=re.DOTALL)
    text = re.sub(r"<local-command-.*?>.*?</local-command-.*?>", "", text,
                  flags=re.DOTALL)
    text = re.sub(r"<command-.*?>.*?</command-.*?>", "", text,
                  flags=re.DOTALL)
    return text.strip()


def _summarize_tool_call(name: str, input_dict: dict) -> str:
    """Create a brief summary of a tool call."""
    if name == "Read":
        path = input_dict.get("file_path", "?")
        return f"Read: {os.path.basename(path)}"
    elif name == "Write":
        path = input_dict.get("file_path", "?")
        return f"Write: {os.path.basename(path)}"
    elif name == "Edit":
        path = input_dict.get("file_path", "?")
        old = input_dict.get("old_string", "")[:50]
        return f"Edit: {os.path.basename(path)} ({old}...)"
    elif name == "Bash":
        cmd = input_dict.get("command", "?")[:TOOL_SUMMARY_MAX]
        return f"Bash: {cmd}"
    elif name == "Grep":
        pattern = input_dict.get("pattern", "?")
        path = input_dict.get("path", "")
        return f"Grep: '{pattern}' in {os.path.basename(path) if path else '.'}"
    elif name == "Glob":
        return f"Glob: {input_dict.get('pattern', '?')}"
    elif name == "Agent":
        desc = input_dict.get("description", "?")
        return f"Agent: {desc}"
    elif name == "TaskCreate":
        return f"Task: {input_dict.get('subject', '?')}"
    elif name == "TaskUpdate":
        return f"TaskUpdate: #{input_dict.get('taskId', '?')} → {input_dict.get('status', '?')}"
    else:
        return f"{name}: {str(input_dict)[:TOOL_SUMMARY_MAX]}"


def format_conversation_md(entries: list[dict], session_id: str,
                           jsonl_path: str) -> str:
    """Format conversation entries as readable markdown."""
    lines = []

    # Header
    first_ts = entries[0]["timestamp"] if entries else ""
    last_ts = entries[-1]["timestamp"] if entries else ""
    date_str = first_ts[:10] if first_ts else "unknown"

    lines.append(f"# Conversation: {date_str} — Session {session_id[:8]}")
    lines.append("")
    lines.append(f"> **Source:** `{os.path.basename(jsonl_path)}`")
    lines.append(f"> **Started:** {first_ts}")
    lines.append(f"> **Ended:** {last_ts}")
    lines.append(f"> **Messages:** {sum(1 for e in entries if e['role'] == 'human')} human, "
                 f"{sum(1 for e in entries if e['role'] == 'claude')} claude")
    lines.append("")
    lines.append("---")
    lines.append("")

    for entry in entries:
        role = entry["role"]
        content = entry["content"]
        ts = entry["timestamp"]
        tools = entry.get("metadata", {}).get("tool_calls", [])

        time_str = ts[11:19] if len(ts) > 19 else ""

        if role == "human":
            lines.append(f"## Human ({time_str})")
            lines.append("")
            lines.append(content)
            lines.append("")

        elif role == "claude":
            lines.append(f"## Claude ({time_str})")
            lines.append("")
            if tools:
                lines.append(f"*Tools used: {len(tools)}*")
                for t in tools[:10]:  # Limit to 10 tool summaries
                    lines.append(f"- `{t}`")
                if len(tools) > 10:
                    lines.append(f"- *... and {len(tools) - 10} more*")
                lines.append("")
            if content:
                lines.append(content)
            lines.append("")

    return "\n".join(lines)


def get_session_id_from_path(path: str) -> str:
    """Extract session ID from JSONL filename."""
    basename = os.path.basename(path)
    return basename.replace(".jsonl", "")


def get_session_date(jsonl_path: str) -> str:
    """Get date from first entry in JSONL."""
    try:
        with open(jsonl_path) as f:
            for line in f:
                d = json.loads(line)
                ts = d.get("timestamp", "")
                if ts:
                    return ts[:10].replace("-", "")
    except Exception:
        pass
    return datetime.now().strftime("%Y%m%d")


def find_latest_jsonl() -> str:
    """Find the most recently modified JSONL file."""
    jsonls = list(Path(JSONL_DIR).glob("*.jsonl"))
    if not jsonls:
        return ""
    return str(max(jsonls, key=lambda p: p.stat().st_mtime))


def find_all_jsonls() -> list[str]:
    """Find all JSONL files sorted by modification time."""
    jsonls = list(Path(JSONL_DIR).glob("*.jsonl"))
    return [str(p) for p in sorted(jsonls, key=lambda p: p.stat().st_mtime)]


def is_already_parsed(jsonl_path: str) -> bool:
    """Check if this JSONL has already been parsed (output exists)."""
    session_id = get_session_id_from_path(jsonl_path)
    for f in Path(OUTPUT_DIR).glob(f"*{session_id[:8]}*"):
        return True
    return False


def parse_and_save(jsonl_path: str, force: bool = False) -> str:
    """Parse JSONL and save as markdown. Returns output path."""
    session_id = get_session_id_from_path(jsonl_path)
    date_str = get_session_date(jsonl_path)

    if not force and is_already_parsed(jsonl_path):
        print(f"Already parsed: {session_id[:8]}")
        return ""

    print(f"Parsing: {os.path.basename(jsonl_path)} ({os.path.getsize(jsonl_path) / 1024 / 1024:.1f} MB)")

    entries = parse_jsonl_to_conversation(jsonl_path)
    if not entries:
        print(f"  No conversation entries found")
        return ""

    md = format_conversation_md(entries, session_id, jsonl_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR,
                                f"CONVERSATION_{date_str}_{session_id[:8]}.md")
    with open(output_path, "w") as f:
        f.write(md)

    human_count = sum(1 for e in entries if e["role"] == "human")
    claude_count = sum(1 for e in entries if e["role"] == "claude")
    print(f"  Saved: {output_path} ({human_count} human, {claude_count} claude messages)")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Parse Claude Code JSONL sessions into readable conversations")
    parser.add_argument("--file", help="Specific JSONL file to parse")
    parser.add_argument("--all", action="store_true",
                        help="Parse all JSONL files")
    parser.add_argument("--auto", action="store_true",
                        help="Auto mode: parse latest, skip if already done")
    parser.add_argument("--force", action="store_true",
                        help="Force re-parse even if output exists")
    args = parser.parse_args()

    if args.file:
        parse_and_save(args.file, force=args.force)

    elif args.all:
        for jsonl in find_all_jsonls():
            parse_and_save(jsonl, force=args.force)

    elif args.auto:
        latest = find_latest_jsonl()
        if latest:
            parse_and_save(latest, force=False)

    else:
        # Default: parse latest
        latest = find_latest_jsonl()
        if latest:
            parse_and_save(latest, force=True)
        else:
            print("No JSONL files found")


if __name__ == "__main__":
    main()
