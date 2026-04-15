"""
channels/commands.py — Unified command registry for all Titan comm channels.

Provides a single source of truth for slash commands available across
Telegram, Discord, Slack, WhatsApp, and Terminal channels.
Each command handler returns a plain string (or list of strings) that
the channel adapter formats for its platform.
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Awaitable, Optional

import httpx

logger = logging.getLogger(__name__)

# Project root (titan_plugin/../)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Command:
    """A registered slash command."""
    name: str
    description: str
    handler: Callable[..., Awaitable[str]]
    subcommands: list[str] = field(default_factory=list)
    maker_only: bool = False


class CommandRegistry:
    """
    Unified command handler for all Titan communication channels.

    Usage:
        registry = CommandRegistry(titan_base_url="http://127.0.0.1:7777")
        result = await registry.execute("/status", user_id="tg_123")
    """

    def __init__(self, titan_base_url: str = "http://127.0.0.1:7777"):
        self.titan_url = titan_base_url.rstrip("/")
        self._commands: dict[str, Command] = {}
        self._maker_ids: set[str] = set()
        self._register_builtins()
        self._load_maker_ids()

    def _load_maker_ids(self) -> None:
        """Load registered maker platform IDs from config."""
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                import tomli as tomllib  # type: ignore[no-redef]
            config_path = _PROJECT_ROOT / "titan_plugin" / "config.toml"
            if config_path.exists():
                with open(config_path, "rb") as f:
                    cfg = tomllib.load(f)
                maker_ids = cfg.get("channels", {}).get("maker_platform_ids", "")
                if maker_ids:
                    self._maker_ids = {mid.strip() for mid in maker_ids.split(",") if mid.strip()}
        except Exception:
            pass

    def _register_builtins(self) -> None:
        """Register all built-in commands."""
        self._commands = {
            "commands": Command(
                name="commands",
                description="List all available commands",
                handler=self._cmd_commands,
            ),
            "status": Command(
                name="status",
                description="Show Titan's cognitive state",
                handler=self._cmd_status,
            ),
            "health": Command(
                name="health",
                description="Quick health check",
                handler=self._cmd_health,
            ),
            "mood": Command(
                name="mood",
                description="Current mood label",
                handler=self._cmd_mood,
            ),
            "wallet": Command(
                name="wallet",
                description="Show wallet balance and address",
                handler=self._cmd_wallet,
            ),
            "birth": Command(
                name="birth",
                description="Genesis NFT data and age",
                handler=self._cmd_birth,
            ),
            "maker": Command(
                name="maker",
                description="Maker identifying info",
                handler=self._cmd_maker,
            ),
            "soul": Command(
                name="soul",
                description="View titan.md sections",
                handler=self._cmd_soul,
                subcommands=["genesis", "nature", "directives", "capabilities",
                             "aspirations", "relationships", "chronicle"],
            ),
            "settings": Command(
                name="settings",
                description="View/change config settings",
                handler=self._cmd_settings,
                maker_only=True,
            ),
            "install": Command(
                name="install",
                description="Install a skill or MCP server",
                handler=self._cmd_install,
                maker_only=True,
            ),
            "uninstall": Command(
                name="uninstall",
                description="Uninstall a skill",
                handler=self._cmd_uninstall,
                maker_only=True,
            ),
            "restart": Command(
                name="restart",
                description="Restart Titan (maker only)",
                handler=self._cmd_restart,
                maker_only=True,
            ),
            "trinity": Command(
                name="trinity",
                description="V3 Divine Trinity tensors (Body/Mind/Spirit)",
                handler=self._cmd_trinity,
            ),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_command(self, text: str) -> bool:
        """Check if text starts with a slash command."""
        if not text or not text.startswith("/"):
            return False
        cmd_name = text.split()[0].lstrip("/").lower()
        return cmd_name in self._commands

    async def execute(self, text: str, user_id: str = "") -> str:
        """
        Parse and execute a slash command.

        Args:
            text: Full command string (e.g., "/soul genesis")
            user_id: Platform-prefixed user ID for auth (e.g., "tg_12345")

        Returns:
            Response string to send back to user.
        """
        parts = text.strip().split(maxsplit=1)
        cmd_name = parts[0].lstrip("/").lower()
        args = parts[1] if len(parts) > 1 else ""

        cmd = self._commands.get(cmd_name)
        if not cmd:
            return f"Unknown command: /{cmd_name}\nType /commands for available commands."

        if cmd.maker_only and user_id not in self._maker_ids:
            return f"/{cmd_name} is restricted to the Maker."

        try:
            return await cmd.handler(args, user_id)
        except Exception as e:
            logger.error("[Commands] /%s failed: %s", cmd_name, e)
            return f"Command failed: {e}"

    def get_command_list(self) -> list[tuple[str, str]]:
        """Return list of (name, description) for autocomplete registration."""
        return [(c.name, c.description) for c in self._commands.values()
                if not c.maker_only]

    def get_all_commands(self) -> list[tuple[str, str]]:
        """Return all commands including maker-only."""
        return [(c.name, c.description) for c in self._commands.values()]

    def get_autocomplete(self, cmd_name: str) -> list[str]:
        """Return subcommand suggestions for a given command."""
        cmd = self._commands.get(cmd_name)
        return cmd.subcommands if cmd else []

    # ------------------------------------------------------------------
    # Command Handlers
    # ------------------------------------------------------------------

    async def _cmd_commands(self, args: str, user_id: str) -> str:
        lines = ["-- Titan Commands --"]
        for cmd in self._commands.values():
            marker = " [maker]" if cmd.maker_only else ""
            lines.append(f"  /{cmd.name} — {cmd.description}{marker}")
        return "\n".join(lines)

    async def _cmd_status(self, args: str, user_id: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(f"{self.titan_url}/status")
            if resp.status_code == 200:
                body = resp.json()
                d = body.get("data", body)  # _ok() wraps in {"status":"ok","data":{...}}
                mood = d.get("mood", {})
                mood_label = mood.get("label", "?") if isinstance(mood, dict) else mood
                energy = d.get("energy_state", {})
                energy_label = energy.get("label", str(energy)) if isinstance(energy, dict) else energy
                nodes = d.get("persistent_nodes", d.get("memory_nodes", "?"))
                uptime_s = d.get("uptime_seconds", 0)
                sol = d.get("sol_balance", "?")
                sov = d.get("sovereignty_pct", "?")

                # Format uptime
                if isinstance(uptime_s, (int, float)) and uptime_s > 0:
                    hours, rem = divmod(int(uptime_s), 3600)
                    mins = rem // 60
                    uptime_str = f"{hours}h {mins}m"
                else:
                    uptime_str = "?"

                return (
                    "-- Titan Status --\n"
                    f"Mood: {mood_label}\n"
                    f"Energy: {energy_label}\n"
                    f"Memory nodes: {nodes}\n"
                    f"SOL: {sol}\n"
                    f"Sovereignty: {sov}\n"
                    f"Uptime: {uptime_str}"
                )
            return f"Status unavailable (HTTP {resp.status_code})."
        except Exception as e:
            return f"Cannot reach Titan: {e}"

    async def _cmd_trinity(self, args: str, user_id: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.titan_url}/v3/trinity")
            if resp.status_code == 200:
                body = resp.json()
                d = body.get("data", body)
                trinity = d.get("trinity", {})
                bus = d.get("bus_stats", {})

                body_t = trinity.get("body", {})
                mind_t = trinity.get("mind", {})
                spirit_t = trinity.get("spirit", {})

                body_v = body_t.get("values", [])
                mind_v = mind_t.get("values", [])
                spirit_v = spirit_t.get("values", [])

                body_dims = body_t.get("dims", [])
                mind_dims = mind_t.get("dims", [])
                spirit_dims = spirit_t.get("dims", [])

                lines = ["-- Divine Trinity --\n"]

                lines.append("Body (5DT):")
                for dim, val in zip(body_dims, body_v):
                    bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
                    lines.append(f"  {dim:16s} {bar} {val:.2f}")

                lines.append("\nMind (5DT):")
                for dim, val in zip(mind_dims, mind_v):
                    bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
                    lines.append(f"  {dim:16s} {bar} {val:.2f}")

                lines.append("\nSpirit (3DT+2):")
                for dim, val in zip(spirit_dims, spirit_v):
                    bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
                    lines.append(f"  {dim:16s} {bar} {val:.2f}")

                lines.append(f"\nBus: {bus.get('published', 0)} published, {bus.get('dropped', 0)} dropped")
                return "\n".join(lines)
            return "Trinity endpoint unavailable (V3 mode required)."
        except Exception as e:
            return f"Trinity check failed: {e}"

    async def _cmd_health(self, args: str, user_id: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.titan_url}/health")
            return "Titan is healthy." if resp.status_code == 200 else f"HTTP {resp.status_code}"
        except Exception as e:
            return f"Health check failed: {e}"

    async def _cmd_mood(self, args: str, user_id: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.titan_url}/status/mood")
            if resp.status_code == 200:
                body = resp.json()
                d = body.get("data", body)
                label = d.get("mood_label", "Unknown")
                score = d.get("current_score", "?")
                delta = d.get("mood_delta", 0)
                delta_str = f"+{delta}" if isinstance(delta, (int, float)) and delta > 0 else str(delta)
                return f"Current mood: {label} ({score}, {delta_str})"
            return "Mood endpoint unavailable."
        except Exception as e:
            return f"Cannot fetch mood: {e}"

    async def _cmd_wallet(self, args: str, user_id: str) -> str:
        try:
            # Load wallet info from config + Solana RPC
            try:
                import tomllib
            except ModuleNotFoundError:
                import tomli as tomllib  # type: ignore[no-redef]

            config_path = _PROJECT_ROOT / "titan_plugin" / "config.toml"
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)

            network_cfg = cfg.get("network", {})
            solana_network = network_cfg.get("solana_network", "devnet")

            # Get pubkey from genesis record
            genesis_path = _PROJECT_ROOT / "data" / "genesis_record.json"
            if genesis_path.exists():
                with open(genesis_path) as f:
                    genesis = json.load(f)
                pubkey = genesis.get("titan_pubkey", "Unknown")
            else:
                pubkey = "Not yet born (no genesis record)"

            # Fetch balance from Solana RPC
            balance_str = "Unknown"
            try:
                rpc_url = (
                    "https://api.devnet.solana.com" if solana_network == "devnet"
                    else "https://api.mainnet-beta.solana.com"
                )
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(rpc_url, json={
                        "jsonrpc": "2.0", "id": 1,
                        "method": "getBalance",
                        "params": [pubkey],
                    })
                if resp.status_code == 200:
                    result = resp.json().get("result", {})
                    lamports = result.get("value", 0)
                    balance_str = f"{lamports / 1e9:.4f} SOL"
            except Exception:
                balance_str = "Could not fetch balance"

            return (
                "-- Titan Wallet --\n"
                f"Network: {solana_network}\n"
                f"Address: {pubkey}\n"
                f"Balance: {balance_str}"
            )
        except Exception as e:
            return f"Wallet info unavailable: {e}"

    async def _cmd_birth(self, args: str, user_id: str) -> str:
        try:
            genesis_path = _PROJECT_ROOT / "data" / "genesis_record.json"
            if not genesis_path.exists():
                return "Titan has not yet been born (no genesis record)."

            with open(genesis_path) as f:
                genesis = json.load(f)

            birth_ts = genesis.get("genesis_time", 0)
            birth_dt = datetime.fromtimestamp(birth_ts, tz=timezone.utc)
            now = datetime.now(timezone.utc)
            age = now - birth_dt
            days = age.days
            hours = age.seconds // 3600

            genesis_tx = genesis.get("genesis_tx", "")
            pubkey = genesis.get("titan_pubkey", "Unknown")

            lines = [
                "-- Titan Birth Record --",
                f"Born: {birth_dt.strftime('%Y-%m-%d %H:%M UTC')}",
                f"Age: {days} days, {hours} hours",
                f"Generation: {genesis.get('version', '?')}",
                f"Identity: {pubkey}",
            ]
            if genesis_tx:
                lines.append(f"Genesis TX: {genesis_tx}")

            return "\n".join(lines)
        except Exception as e:
            return f"Birth data unavailable: {e}"

    async def _cmd_maker(self, args: str, user_id: str) -> str:
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                import tomli as tomllib  # type: ignore[no-redef]

            config_path = _PROJECT_ROOT / "titan_plugin" / "config.toml"
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)

            network = cfg.get("network", {})
            twitter = cfg.get("twitter_social", {})

            maker_pubkey = network.get("maker_pubkey", "Unknown")
            x_handle = twitter.get("user_name", "Not configured")

            return (
                "-- Maker Info --\n"
                f"Wallet: {maker_pubkey}\n"
                f"X/Twitter: @{x_handle}"
            )
        except Exception as e:
            return f"Maker info unavailable: {e}"

    async def _cmd_soul(self, args: str, user_id: str) -> str:
        try:
            soul_path = _PROJECT_ROOT / "titan.md"
            if not soul_path.exists():
                return "titan.md not found."

            content = soul_path.read_text(encoding="utf-8")

            # Strip YAML frontmatter
            if content.startswith("---"):
                end = content.find("---", 3)
                if end > 0:
                    content = content[end + 3:].strip()

            if not args:
                # List available sections
                sections = []
                for line in content.split("\n"):
                    if line.startswith("## "):
                        sections.append(line[3:].strip())
                return "-- Soul Sections --\n" + "\n".join(f"  /soul {s.lower()}" for s in sections)

            # Find requested section
            target = args.strip().lower()
            lines = content.split("\n")
            capture = False
            result: list[str] = []

            for line in lines:
                if line.startswith("## "):
                    if capture:
                        break  # End of target section
                    if target in line.lower():
                        capture = True
                        result.append(line)
                        continue
                if capture:
                    result.append(line)

            if result:
                text = "\n".join(result).strip()
                return text[:3000]  # Cap for channel limits
            return f"Section '{args}' not found. Type /soul for available sections."
        except Exception as e:
            return f"Soul read failed: {e}"

    async def _cmd_settings(self, args: str, user_id: str) -> str:
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                import tomli as tomllib  # type: ignore[no-redef]

            config_path = _PROJECT_ROOT / "titan_plugin" / "config.toml"
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)

            if not args:
                # List top-level sections
                sections = list(cfg.keys())
                return "-- Settings Sections --\n" + "\n".join(
                    f"  /settings {s}" for s in sections
                )

            parts = args.strip().split(maxsplit=1)
            section_name = parts[0]

            if section_name not in cfg:
                return f"Unknown section: {section_name}\nType /settings for available sections."

            section = cfg[section_name]

            if len(parts) == 1:
                # Show section contents (mask sensitive keys)
                lines = [f"-- [{section_name}] --"]
                for k, v in section.items():
                    display_v = _mask_sensitive(k, v)
                    lines.append(f"  {k} = {display_v}")
                return "\n".join(lines)

            # Set a value: /settings section key=value
            kv = parts[1]
            if "=" not in kv:
                # Show single key
                key = kv.strip()
                if key in section:
                    return f"[{section_name}] {key} = {_mask_sensitive(key, section[key])}"
                return f"Key '{key}' not found in [{section_name}]."

            # Update value
            key, value = kv.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key not in section:
                return f"Key '{key}' not found in [{section_name}]."

            # Type-coerce based on existing value
            old_val = section[key]
            try:
                if isinstance(old_val, bool):
                    new_val = value.lower() in ("true", "1", "yes")
                elif isinstance(old_val, int):
                    new_val = int(value)
                elif isinstance(old_val, float):
                    new_val = float(value)
                else:
                    new_val = value
            except ValueError:
                return f"Invalid value type for {key}. Expected {type(old_val).__name__}."

            # Write back using toml update (read raw, replace line)
            raw = config_path.read_text(encoding="utf-8")
            import re
            # Match the key in the correct section
            pattern = re.compile(
                rf'^(\s*{re.escape(key)}\s*=\s*).*$',
                re.MULTILINE,
            )
            if isinstance(new_val, str):
                replacement = f'\\1"{new_val}"'
            elif isinstance(new_val, bool):
                replacement = f"\\1{'true' if new_val else 'false'}"
            else:
                replacement = f"\\1{new_val}"

            new_raw = pattern.sub(replacement, raw, count=1)
            config_path.write_text(new_raw, encoding="utf-8")

            # Signal hot-reload
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(f"{self.titan_url}/maker/reload-config",
                                      headers={"X-Titan-Internal-Key": _get_internal_key()})
            except Exception:
                pass  # Best-effort reload

            return f"Updated [{section_name}] {key} = {_mask_sensitive(key, new_val)}"

        except Exception as e:
            return f"Settings error: {e}"

    async def _cmd_install(self, args: str, user_id: str) -> str:
        if not args:
            return "Usage: /install <url_or_path>\nInstalls a skill or MCP server."
        try:
            from titan_plugin.skills.registry import SkillRegistry
            from titan_plugin.skills.validator import SkillValidator
            from titan_plugin.skills.installer import SkillInstaller

            registry = SkillRegistry()  # ~/.titan/skills/
            validator = SkillValidator()  # Static-only (no guardian/LLM in channel process)
            installer = SkillInstaller(registry=registry, validator=validator)

            result = await installer.install(args.strip(), force=False)

            if result.requires_confirmation:
                # Auto-force for maker (they already passed auth)
                result = await installer.install(args.strip(), force=True)

            if result.success:
                risk_info = ""
                if result.validation:
                    risk_info = f" (safety: {result.validation.risk_level}, score: {result.validation.risk_score})"
                return f"Installed: {result.skill_name}{risk_info}\nSaved to: {result.file_path}"
            return f"Install failed: {result.error}"
        except Exception as e:
            return f"Install failed: {e}"

    async def _cmd_uninstall(self, args: str, user_id: str) -> str:
        if not args:
            return "Usage: /uninstall <skill_name>"
        try:
            from titan_plugin.skills.registry import SkillRegistry

            registry = SkillRegistry()  # ~/.titan/skills/
            registry.load_all()  # Load existing skills so we can find by name
            name = args.strip()
            # Strip .md extension if user included it (skills are indexed by name, not filename)
            if name.endswith(".md"):
                name = name[:-3]

            if registry.remove_skill(name):
                return f"Uninstalled: {name}"
            return f"Skill '{name}' not found. Installed skills: {', '.join(s.name for s in registry.list_skills()) or 'none'}"
        except Exception as e:
            return f"Uninstall failed: {e}"

    async def _cmd_restart(self, args: str, user_id: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    "http://127.0.0.1:7778/restart",
                    json={"requester": user_id},
                )
            if resp.status_code == 200:
                # Store callback info for post-restart notification
                self._restart_pending = True
                self._restart_user_id = user_id
                # Start background health poll
                import asyncio
                asyncio.create_task(self._poll_restart_health())
                return "Titan restart initiated. Will notify you when back online (~30s)..."
            return f"Restart failed (HTTP {resp.status_code})."
        except httpx.ConnectError:
            return "Watchdog not running. Restart manually via SSH."
        except Exception as e:
            return f"Restart failed: {e}"

    async def _poll_restart_health(self) -> None:
        """Poll Titan health after restart and store result."""
        import asyncio
        for i in range(12):  # 12 x 5s = 60s max wait
            await asyncio.sleep(5)
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{self.titan_url}/health")
                if resp.status_code == 200:
                    self._restart_result = "Titan is back online!"
                    self._restart_pending = False
                    return
            except Exception:
                continue
        self._restart_result = "Titan did not come back within 60s. Check logs."
        self._restart_pending = False


# ------------------------------------------------------------------
# Info line formatter (status bar for responses)
# ------------------------------------------------------------------

def format_info_line(result: dict) -> str:
    """
    Format a compact status line to prepend to Titan's response.
    Shows mode, mood, and energy in a single line.
    """
    if "error" in result:
        mode = result.get("mode", "Error")
        return f"[{mode}]"

    mode = result.get("mode", "?")
    mood = result.get("mood", "?")
    return f"[{mode} | {mood}]"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_SENSITIVE_KEYS = {
    "api_key", "bot_token", "app_token", "token", "password", "secret",
    "internal_key", "privy_api_key", "venice_api_key", "openrouter_api_key",
    "ollama_cloud_api_key", "firecrawl_api_key", "twitterapi_io_key",
    "webshare_rotating_url", "totp_secret", "venice_client_cookie",
    "auth_session",
}


def _mask_sensitive(key: str, value) -> str:
    """Mask sensitive config values for display."""
    key_lower = key.lower()
    if any(s in key_lower for s in ("key", "token", "password", "secret", "cookie", "session")):
        s = str(value)
        if len(s) > 8:
            return f'"{s[:4]}...{s[-4:]}"'
        elif s:
            return '"****"'
    return repr(value)


def _get_internal_key() -> str:
    """Load internal API key from config.toml."""
    try:
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]
        config_path = _PROJECT_ROOT / "titan_plugin" / "config.toml"
        if config_path.exists():
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)
            return cfg.get("api", {}).get("internal_key", "")
    except Exception:
        pass
    return ""
