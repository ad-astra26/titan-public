#!/usr/bin/env python3
"""Decisive agno-path diagnostic: does agno's OpenAILike → Ollama Cloud emit a
tool call for gemma4? Mirrors production transport (openai SDK via agno),
NOT raw httpx. If raw httpx works but this does not, the bug is in agno's
schema-gen / tool_choice / response parsing."""
import asyncio
import logging
import sys
import tomllib
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
# Surface the actual outgoing request the openai SDK makes
logging.getLogger("openai").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("httpcore").setLevel(logging.INFO)

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
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:31b"

from agno.agent import Agent
from agno.models.openai.like import OpenAILike


async def get_weather(city: str) -> str:
    """Get the current weather for a city. ALWAYS use this when asked about weather.

    Args:
        city: The city name.
    """
    return f"WEATHER_TOOL_RAN: {city} is 12C and rainy."


async def main():
    model = OpenAILike(id=MODEL, name="OllamaCloud", api_key=API_KEY, base_url=BASE_URL)
    agent = Agent(
        name="DiagAgent",
        model=model,
        instructions=[
            "You are a helpful assistant.",
            "When asked about weather you MUST call the get_weather tool. Never answer from memory.",
        ],
        tools=[get_weather],
        telemetry=False,
        tool_call_limit=3,
        markdown=True,
    )
    print(f"\n{'='*70}\nAGNO PATH — model={MODEL}\n{'='*70}")
    out = await agent.arun("What's the weather in Prague right now?")
    print(f"\n--- RESULT ---")
    content = getattr(out, "content", out)
    print(f"content: {content}")
    # Inspect run for tool calls
    tools_used = []
    for attr in ("tools", "messages"):
        val = getattr(out, attr, None)
        if val:
            print(f"\nrun_output.{attr}: {val}")
    ran = "WEATHER_TOOL_RAN" in str(content)
    print(f"\n>>> TOOL ACTUALLY EXECUTED (sentinel in output): {ran}")


asyncio.run(main())
