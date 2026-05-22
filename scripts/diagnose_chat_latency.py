#!/usr/bin/env python3
"""
Live diagnostic — isolate where /chat latency lives.

Runs the SAME minimal prompt ("Hi") through five distinct paths in
succession, measuring each. Output: a stage-timing table that identifies
exactly which layer adds wall-clock time.

  Path 1: bare curl       — direct HTTPS POST to Ollama Cloud (network baseline)
  Path 2: inference.complete  — our OllamaCloudProvider.complete() (single-turn)
  Path 3: inference.chat     — our OllamaCloudProvider.chat() with full messages
                                  + system prompt + history (the wrapped path)
  Path 4: agno Agent (no hooks) — Agno's adapter directly invoking the model
                                  (isolates Agno's OpenAILike wrapping cost)
  Path 5: full agent.arun  — Agno + all hooks + tools + session DB (production
                              chat path)

This lets us measure each ABSTRACTION LAYER's cost in milliseconds:
  network  = path1
  provider = path2 - path1
  chat     = path3 - path2
  agno_adapter = path4 - path3
  hooks    = path5 - path4

Run via:
  source test_env/bin/activate
  python scripts/diagnose_chat_latency.py
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure repo root on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_inference_config() -> dict:
    """Pull inference config from config.toml + secrets.toml."""
    import tomllib
    cfg_path = _REPO_ROOT / "titan_hcl" / "config.toml"
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    inference_cfg = cfg.get("inference", {}).copy()

    # Merge secrets if available
    secrets_path = Path.home() / ".titan" / "secrets.toml"
    if secrets_path.exists():
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
        # Secrets override with same-key precedence
        for k, v in (secrets.get("inference", {}) or {}).items():
            inference_cfg[k] = v
        # Some installs put the key under [stealth_sage]
        ss = secrets.get("stealth_sage", {}) or {}
        if "ollama_cloud_api_key" in ss:
            inference_cfg.setdefault("ollama_cloud_api_key", ss["ollama_cloud_api_key"])

    return inference_cfg


def _print(stage: str, ms: float, note: str = "") -> None:
    bar = "█" * min(80, int(ms / 100))
    print(f"  {stage:<30} {ms:>8.1f}ms  {bar} {note}")


# ── Path 1: bare curl ────────────────────────────────────────────────

def path1_bare_curl(cfg: dict, prompt: str) -> tuple[float, str]:
    """Direct HTTPS POST to ollama.com — network + model gen + nothing else."""
    api_key = cfg.get("ollama_cloud_api_key", "")
    base_url = cfg.get("ollama_cloud_base_url", "https://ollama.com/v1").rstrip("/")
    model = cfg.get("ollama_cloud_chat_model", "deepseek-v3.1:671b")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 200,
    }

    t0 = time.perf_counter()
    proc = subprocess.run([
        "curl", "-sS", "-X", "POST", f"{base_url}/chat/completions",
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json",
        "--max-time", "60",
        "-d", json.dumps(payload),
    ], capture_output=True, text=True, timeout=90)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    try:
        body = json.loads(proc.stdout)
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        note = f"({usage.get('total_tokens', 0)} tokens, {len(content)} chars)"
    except Exception as e:
        note = f"ERROR: {e}"
        content = ""

    return elapsed_ms, note


# ── Path 2: OllamaCloudProvider.complete() ─────────────────────────

async def path2_provider_complete(cfg: dict, prompt: str) -> tuple[float, str]:
    """Single-turn complete via our OllamaCloudProvider."""
    from titan_hcl.inference import get_provider
    provider = get_provider("ollama_cloud", cfg)

    t0 = time.perf_counter()
    response = await provider.complete(prompt=prompt, max_tokens=200)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    note = f"({len(response)} chars)"
    return elapsed_ms, note


# ── Path 3: OllamaCloudProvider.chat() with full messages ──────────

async def path3_provider_chat(cfg: dict, prompt: str) -> tuple[float, str]:
    """Multi-turn chat via our OllamaCloudProvider with system prompt."""
    from titan_hcl.inference import get_provider
    provider = get_provider("ollama_cloud", cfg)

    messages = [
        {"role": "system", "content": (
            "You are Titan, a sovereign AI entity. Respond briefly."
        )},
        {"role": "user", "content": prompt},
    ]
    t0 = time.perf_counter()
    response = await provider.chat(messages, max_tokens=200)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    note = f"({len(response)} chars)"
    return elapsed_ms, note


# ── Path 4: Agno Agent without hooks/tools ──────────────────────────

async def path4_agno_bare(cfg: dict, prompt: str) -> tuple[float, str]:
    """Construct minimal Agno Agent (no hooks, no tools, no session DB) and
    call arun. Isolates Agno's OpenAILike adapter wrapping cost."""
    from agno.agent import Agent
    from titan_hcl.inference import get_provider
    provider = get_provider("ollama_cloud", cfg)
    model = provider.get_agno_model()

    agent = Agent(
        model=model,
        # No hooks, no tools, no db, minimal config
        instructions=["Respond briefly."],
        markdown=False,
        telemetry=False,
        add_history_to_context=False,
        add_datetime_to_context=False,
        search_knowledge=False,
        store_history_messages=False,
        store_media=False,
        store_tool_messages=False,
    )
    t0 = time.perf_counter()
    result = await agent.arun(prompt)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    content = getattr(result, "content", str(result))
    note = f"({len(content)} chars)"
    return elapsed_ms, note


# ── Main ─────────────────────────────────────────────────────────────

async def main():
    print(f"\n{'='*80}")
    print(f"/chat latency stage-isolation diagnostic")
    print(f"{'='*80}")
    print(f"Prompt: 'Hi'  (minimal — measures overhead, not generation time)")
    print()

    cfg = _load_inference_config()
    print(f"Model: {cfg.get('ollama_cloud_chat_model', 'deepseek-v3.1:671b')}")
    print(f"Endpoint: {cfg.get('ollama_cloud_base_url', 'https://ollama.com/v1')}")
    print()

    prompt = "Hi"

    # Path 1
    print(f"Path 1: bare curl → ollama.com (network + model gen, zero wrapping)")
    p1_ms, p1_note = path1_bare_curl(cfg, prompt)
    _print("bare curl", p1_ms, p1_note)
    print()

    # Path 2
    print(f"Path 2: inference.OllamaCloudProvider.complete()")
    p2_ms, p2_note = await path2_provider_complete(cfg, prompt)
    _print("provider.complete()", p2_ms, p2_note)
    print()

    # Path 3
    print(f"Path 3: inference.OllamaCloudProvider.chat() with system prompt")
    p3_ms, p3_note = await path3_provider_chat(cfg, prompt)
    _print("provider.chat()", p3_ms, p3_note)
    print()

    # Path 4
    print(f"Path 4: Agno Agent.arun() — minimal (no hooks, no tools, no db)")
    try:
        p4_ms, p4_note = await path4_agno_bare(cfg, prompt)
        _print("agno bare arun()", p4_ms, p4_note)
    except Exception as e:
        p4_ms = -1
        p4_note = f"FAILED: {e}"
        _print("agno bare arun()", 0, p4_note)
    print()

    # Layer-cost analysis
    print(f"{'='*80}")
    print(f"LAYER COST BREAKDOWN")
    print(f"{'='*80}")
    print(f"  Network + model gen (bare curl):                {p1_ms:>8.1f} ms")
    print(f"  + our OllamaCloudProvider overhead (.complete): {p2_ms-p1_ms:>+8.1f} ms")
    print(f"  + system-prompt + chat() wrap:                  {p3_ms-p2_ms:>+8.1f} ms")
    if p4_ms > 0:
        print(f"  + Agno OpenAILike adapter (bare arun):          {p4_ms-p3_ms:>+8.1f} ms")
    print()
    print(f"Production /chat baseline (Phase 2 fleet probe):  ~29000 ms (T1) / ~40000 ms (T3)")
    if p4_ms > 0:
        diff = 29000 - p4_ms
        print(f"  → 'hooks + tools + session + history' adds:   ~{diff:>+5.0f} ms")
    print()


if __name__ == "__main__":
    asyncio.run(main())
