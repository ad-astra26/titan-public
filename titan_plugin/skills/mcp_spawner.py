"""
skills/mcp_spawner.py — MCP child process lifecycle manager.

Spawns, monitors, and terminates MCP tool servers defined in skill files.
Each MCP server runs as a subprocess communicating via stdio.
"""
import asyncio
import logging
import os
import signal
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class MCPProcess:
    """Tracks a running MCP server subprocess."""
    skill_name: str
    server_name: str
    command: str
    args: list
    process: Optional[asyncio.subprocess.Process] = None
    pid: Optional[int] = None
    started_at: float = 0.0
    restart_count: int = 0


class MCPSpawner:
    """
    Manages MCP server child processes for installed skills.

    Each skill with an `mcp:` section gets a subprocess spawned.
    Processes are monitored and can be started/stopped/restarted individually.
    """

    def __init__(self):
        self._processes: Dict[str, MCPProcess] = {}

    async def spawn(self, skill_name: str, mcp_config: dict) -> bool:
        """
        Spawn an MCP server subprocess for a skill.

        Args:
            skill_name: Name of the skill that owns this MCP server.
            mcp_config: Dict with keys: name, command, args.

        Returns:
            True if the process was spawned successfully.
        """
        server_name = mcp_config.get("name", skill_name)
        command = mcp_config.get("command", "python")
        args = mcp_config.get("args", [])

        # Check if already running
        if skill_name in self._processes:
            existing = self._processes[skill_name]
            if existing.process and existing.process.returncode is None:
                logger.warning("[MCPSpawner] %s already running (pid=%s)", skill_name, existing.pid)
                return True

        # Resolve command path
        full_args = [command] + [str(a) for a in args]

        try:
            process = await asyncio.create_subprocess_exec(
                *full_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            import time
            mcp_proc = MCPProcess(
                skill_name=skill_name,
                server_name=server_name,
                command=command,
                args=args,
                process=process,
                pid=process.pid,
                started_at=time.time(),
            )
            self._processes[skill_name] = mcp_proc

            logger.info("[MCPSpawner] Spawned %s (pid=%d): %s",
                        server_name, process.pid, " ".join(full_args))
            return True

        except FileNotFoundError:
            logger.error("[MCPSpawner] Command not found: %s", command)
            return False
        except Exception as e:
            logger.error("[MCPSpawner] Failed to spawn %s: %s", skill_name, e)
            return False

    async def stop(self, skill_name: str, timeout: float = 5.0) -> bool:
        """
        Stop an MCP server subprocess.

        Args:
            skill_name: Name of the skill whose MCP server to stop.
            timeout: Seconds to wait for graceful shutdown before SIGKILL.

        Returns:
            True if the process was stopped.
        """
        mcp_proc = self._processes.get(skill_name)
        if not mcp_proc or not mcp_proc.process:
            logger.debug("[MCPSpawner] No process for %s", skill_name)
            return False

        proc = mcp_proc.process
        if proc.returncode is not None:
            # Already dead
            del self._processes[skill_name]
            return True

        # Graceful shutdown: close stdin (MCP servers exit on stdin EOF)
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Force kill
            logger.warning("[MCPSpawner] Force-killing %s (pid=%d)", skill_name, proc.pid)
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass

        del self._processes[skill_name]
        logger.info("[MCPSpawner] Stopped %s", skill_name)
        return True

    async def restart(self, skill_name: str, mcp_config: dict) -> bool:
        """Stop and re-spawn an MCP server."""
        await self.stop(skill_name)
        mcp_proc = self._processes.get(skill_name)
        if mcp_proc:
            mcp_proc.restart_count += 1
        return await self.spawn(skill_name, mcp_config)

    async def stop_all(self):
        """Stop all running MCP server processes."""
        names = list(self._processes.keys())
        for name in names:
            await self.stop(name)
        logger.info("[MCPSpawner] All MCP servers stopped.")

    def is_running(self, skill_name: str) -> bool:
        """Check if an MCP server is currently running."""
        mcp_proc = self._processes.get(skill_name)
        if not mcp_proc or not mcp_proc.process:
            return False
        return mcp_proc.process.returncode is None

    def list_running(self) -> list:
        """List all running MCP servers."""
        running = []
        for name, mcp_proc in self._processes.items():
            if mcp_proc.process and mcp_proc.process.returncode is None:
                running.append({
                    "skill_name": name,
                    "server_name": mcp_proc.server_name,
                    "pid": mcp_proc.pid,
                    "started_at": mcp_proc.started_at,
                    "restart_count": mcp_proc.restart_count,
                })
        return running

    def get_process(self, skill_name: str) -> Optional[MCPProcess]:
        """Get the MCPProcess for a skill."""
        return self._processes.get(skill_name)
