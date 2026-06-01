#!/usr/bin/env python3
"""Test gemma4 tool-calling reliability with the full 11-tool production set
attached (vs the 1-tool isolated test). Repeats N times to measure hit rate."""
import asyncio
import tomllib
from pathlib import Path

with open(Path.home() / ".titan" / "secrets.toml", "rb") as f:
    sec = tomllib.load(f)
def _find(d, key):
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict) and key in v:
            return v[key]
    return None
API_KEY = _find(sec, "ollama_cloud_api_key")
BASE_URL = (_find(sec, "ollama_cloud_base_url") or "https://ollama.com/v1").rstrip("/")
MODEL = "gemma4:31b"

from agno.agent import Agent
from agno.models.openai.like import OpenAILike

CALLED = {"name": None}

async def coding_sandbox(code: str, expected_stdout: str = "", assertion: str = "") -> str:
    """Execute Python code in an AST-validated subprocess sandbox. Use this to
    compute something deterministically (math, data processing, algorithm
    verification) or check a code snippet's correctness.

    Args:
        code: Python source to execute.
        expected_stdout: Optional expected output string.
        assertion: Optional Python boolean expression to assert.
    """
    CALLED["name"] = "coding_sandbox"
    return "OK: stdout=144"

async def research(query: str) -> str:
    """Trigger autonomous web research. Use for factual info beyond training data.
    Args:
        query: The knowledge gap to research."""
    CALLED["name"] = "research"; return "no results"

async def events_teacher(action: str, topic: str = "") -> str:
    """Distill or fetch recent X/Twitter events.
    Args:
        action: 'distill' or 'fetch'.
        topic: Optional topic filter."""
    return "ok"

async def x_research(action: str, content: str = "") -> str:
    """Post to or fetch from X via the gateway.
    Args:
        action: 'post' or 'fetch'.
        content: Post content."""
    return "ok"

async def generate_art(prompt: str) -> str:
    """Generate an image from a text prompt.
    Args:
        prompt: Image description."""
    return "ok"

async def generate_audio(prompt: str) -> str:
    """Generate audio/music from a text prompt.
    Args:
        prompt: Audio description."""
    return "ok"

async def read_buffer(buffer_name: str) -> str:
    """Read an ACT-R cognitive buffer.
    Args:
        buffer_name: The buffer to read."""
    return "ok"

async def write_buffer(buffer_name: str, content: str) -> str:
    """Write to an ACT-R cognitive buffer.
    Args:
        buffer_name: The buffer.
        content: Content to write."""
    return "ok"

async def clear_buffer(buffer_name: str) -> str:
    """Clear an ACT-R cognitive buffer.
    Args:
        buffer_name: The buffer to clear."""
    return "ok"

async def query_retrieval(query: str) -> str:
    """Recall from memory and populate the retrieval buffer.
    Args:
        query: The recall query."""
    return "ok"

async def match_procedural_skill(goal_text: str) -> str:
    """Match a compiled procedural skill to a goal.
    Args:
        goal_text: The goal to match."""
    return "ok"

ALL_TOOLS = [coding_sandbox, research, events_teacher, x_research, generate_art,
             generate_audio, read_buffer, write_buffer, clear_buffer,
             query_retrieval, match_procedural_skill]

PROMPTS = [
    "Can you help me verify this: is the 12th Fibonacci number 144? Please compute it to be sure.",
    "I'm not sure my code is right — can you check that sum([i**2 for i in range(10)]) equals 285?",
    "Please analyze whether 2**10 is exactly 1024 by computing it.",
]

async def main():
    for directive in (False, True):
        instr = ["You are Titan, a sovereign AI entity."]
        if directive:
            instr.append("When a request involves deterministic computation, math, or "
                         "verifying code correctness, you MUST call the coding_sandbox tool.")
        print(f"\n{'='*70}\n{'WITH' if directive else 'NO'} tool directive — 11 tools attached\n{'='*70}")
        for p in PROMPTS:
            CALLED["name"] = None
            model = OpenAILike(id=MODEL, name="OllamaCloud", api_key=API_KEY, base_url=BASE_URL)
            agent = Agent(name="Titan", model=model, instructions=instr,
                          tools=ALL_TOOLS, telemetry=False, tool_call_limit=3, markdown=True)
            out = await agent.arun(p)
            tools = [t.tool_name for t in (getattr(out, "tools", None) or [])]
            print(f"  [{'CALLED:'+str(tools) if tools else 'NO TOOL':<28}] {p[:55]}")

asyncio.run(main())
