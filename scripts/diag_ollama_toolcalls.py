#!/usr/bin/env python3
"""Definitive diagnostic: does Ollama Cloud /chat/completions emit tool_calls?

Hits the raw OpenAI-compatible endpoint with a tools array for each model tier
and reports whether a `tool_calls` field comes back. Answers model-vs-provider
with evidence, not assumption.
"""
import json
import sys
import tomllib
from pathlib import Path

import httpx

SECRETS = Path.home() / ".titan" / "secrets.toml"
with open(SECRETS, "rb") as f:
    sec = tomllib.load(f)
# secrets.toml may nest under [inference]; flatten search
def _find(d, key):
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict) and key in v:
            return v[key]
    return None

API_KEY = _find(sec, "ollama_cloud_api_key")
BASE_URL = (_find(sec, "ollama_cloud_base_url") or "https://ollama.com/v1").rstrip("/")
assert API_KEY, "no ollama_cloud_api_key in secrets.toml"

MODELS = sys.argv[1:] or ["gemma4:31b", "deepseek-v3.1:671b", "gemma3:4b"]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city. ALWAYS use this when asked about weather.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"}
            },
            "required": ["city"],
        },
    },
}]

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. When the user asks about weather, you MUST call the get_weather tool. Do not answer from memory."},
    {"role": "user", "content": "What's the weather in Prague right now? Call the tool."},
]

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

for model in MODELS:
    payload = {
        "model": model,
        "messages": MESSAGES,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 512,
    }
    print(f"\n{'='*70}\nMODEL: {model}\n{'='*70}")
    try:
        resp = httpx.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=120.0)
        print(f"HTTP {resp.status_code}")
        if resp.status_code != 200:
            print(resp.text[:1000])
            continue
        data = resp.json()
        msg = data["choices"][0]["message"]
        finish = data["choices"][0].get("finish_reason")
        tc = msg.get("tool_calls")
        print(f"finish_reason: {finish}")
        print(f"tool_calls present: {bool(tc)}")
        if tc:
            print(f"tool_calls: {json.dumps(tc, indent=2)[:1500]}")
        content = msg.get("content") or ""
        print(f"content ({len(content)} chars): {content[:600]}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

# Second test: tool_choice forced ("required")
print(f"\n\n{'#'*70}\n# FORCED tool_choice='required' test\n{'#'*70}")
for model in MODELS:
    payload = {
        "model": model,
        "messages": MESSAGES,
        "tools": TOOLS,
        "tool_choice": "required",
        "temperature": 0.2,
        "max_tokens": 512,
    }
    print(f"\n--- {model} (forced) ---")
    try:
        resp = httpx.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=120.0)
        print(f"HTTP {resp.status_code}")
        if resp.status_code != 200:
            print(resp.text[:600])
            continue
        data = resp.json()
        msg = data["choices"][0]["message"]
        tc = msg.get("tool_calls")
        print(f"finish_reason: {data['choices'][0].get('finish_reason')} | tool_calls: {bool(tc)}")
        if tc:
            print(json.dumps(tc, indent=2)[:800])
        else:
            print(f"content: {(msg.get('content') or '')[:300]}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
