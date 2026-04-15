#!/usr/bin/env python3
"""
start_channels.py — Launch enabled channel adapters for Titan.

Reads [channels] from config.toml and starts each enabled adapter as an
independent asyncio task.  If one adapter crashes the others continue.

Usage:
    python scripts/start_channels.py [--titan-url http://127.0.0.1:7777]
"""
import argparse
import asyncio
import logging
import pathlib
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("titan.channels")

# Ensure the project root is on sys.path so titan_plugin is importable
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from titan_plugin.channels import get_channel_config  # noqa: E402


# ---------------------------------------------------------------------------
# Adapter registry: channel_name -> (module_path, start_function_name)
# ---------------------------------------------------------------------------
_ADAPTERS = {
    "telegram": ("titan_plugin.channels.telegram", "start_telegram"),
    "discord":  ("titan_plugin.channels.discord_bot", "start_discord"),
    "slack":    ("titan_plugin.channels.slack_bot", "start_slack"),
    "whatsapp": ("titan_plugin.channels.whatsapp", "start_whatsapp"),
}


async def _launch_adapter(name: str, module_path: str, func_name: str, titan_url: str) -> None:
    """Import and run a single adapter, catching all exceptions."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        start_fn = getattr(mod, func_name)
        config = get_channel_config(name)
        logger.info("Starting %s adapter...", name)
        await start_fn(config, titan_url)
    except Exception:
        logger.exception("Channel adapter '%s' crashed.", name)


async def main(titan_url: str) -> None:
    """Discover enabled channels and start them concurrently."""
    # Load the full channels config to check enabled flags
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    config_path = _PROJECT_ROOT / "titan_plugin" / "config.toml"
    if not config_path.exists():
        logger.error("config.toml not found at %s", config_path)
        sys.exit(1)

    with open(config_path, "rb") as fh:
        full_config = tomllib.load(fh)

    channels_section = full_config.get("channels", {})

    tasks: list[asyncio.Task] = []
    for name, (mod_path, func_name) in _ADAPTERS.items():
        if channels_section.get(f"{name}_enabled", False):
            task = asyncio.create_task(
                _launch_adapter(name, mod_path, func_name, titan_url),
                name=f"channel-{name}",
            )
            tasks.append(task)
        else:
            logger.info("Channel '%s' is disabled — skipping.", name)

    if not tasks:
        logger.warning("No channel adapters are enabled. Edit [channels] in config.toml.")
        return

    logger.info("Running %d channel adapter(s). Press Ctrl+C to stop.", len(tasks))

    try:
        # Wait forever (or until all tasks finish / crash)
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Shutting down channel adapters...")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Start Titan channel adapters.")
    parser.add_argument(
        "--titan-url",
        default="http://127.0.0.1:7777",
        help="Base URL of the Titan API (default: http://127.0.0.1:7777)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.titan_url))
    except KeyboardInterrupt:
        logger.info("Interrupted — goodbye.")


if __name__ == "__main__":
    cli()
