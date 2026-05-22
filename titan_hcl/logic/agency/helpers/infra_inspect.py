"""
infra_inspect helper — System metrics + log reader for self-awareness.

Enriches: Body (all 5 dims get real-time update from actual hardware)
Always available (reads local system — no external dependencies).
"""
import logging
import os

logger = logging.getLogger(__name__)


class InfraInspectHelper:
    """Infrastructure inspection: system stats, logs, process info."""

    def __init__(self, log_path: str = "/tmp/titan_agent.log"):
        self._log_path = log_path

    @property
    def name(self) -> str:
        return "infra_inspect"

    @property
    def description(self) -> str:
        return "Inspect system metrics, logs, and running processes"

    @property
    def capabilities(self) -> list[str]:
        return ["system_stats", "log_reader", "process_list"]

    @property
    def resource_cost(self) -> str:
        return "low"

    @property
    def latency(self) -> str:
        return "fast"

    @property
    def enriches(self) -> list[str]:
        return ["body"]

    @property
    def requires_sandbox(self) -> bool:
        return False

    async def execute(self, params: dict) -> dict:
        """
        Execute infrastructure inspection.

        Params:
            what: "system" | "logs" | "processes" (default: "system")
            lines: Number of log lines to return (default: 20)
        """
        what = params.get("what", "system")

        try:
            if what == "system":
                return await self._inspect_system()
            elif what == "logs":
                lines = params.get("lines", 20)
                return self._inspect_logs(lines)
            elif what == "processes":
                return self._inspect_processes()
            else:
                return await self._inspect_system()
        except Exception as e:
            logger.warning("[InfraInspect] Failed: %s", e)
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": str(e)}

    async def _inspect_system(self) -> dict:
        """Get system metrics via psutil."""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            result = (
                f"CPU: {cpu}% | "
                f"RAM: {mem.percent}% ({mem.used // (1024**2)}MB / {mem.total // (1024**2)}MB) | "
                f"Disk: {disk.percent}% ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)"
            )

            # Enrichment: map real metrics to body tensor dimensions
            enrichment = {
                "body": [0, 1, 2, 3, 4],
                "values": {
                    0: 1.0 - (cpu / 100.0),           # interoception (inverse CPU load)
                    2: 1.0 - (mem.percent / 100.0),    # somatosensation (inverse RAM)
                    4: 1.0 - (disk.percent / 100.0),   # thermal (inverse disk)
                },
            }

            return {"success": True, "result": result, "enrichment_data": enrichment,
                    "error": None}
        except ImportError:
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": "psutil not installed"}

    def _inspect_logs(self, lines: int = 20) -> dict:
        """Read recent log lines."""
        if not os.path.exists(self._log_path):
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": f"Log file not found: {self._log_path}"}

        try:
            with open(self._log_path, "r") as f:
                all_lines = f.readlines()
            recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
            result = "".join(recent)
            return {"success": True, "result": result[:500], "enrichment_data": {},
                    "error": None}
        except Exception as e:
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": str(e)}

    def _inspect_processes(self) -> dict:
        """List Titan-related processes."""
        try:
            import psutil
            procs = []
            for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
                try:
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    if "titan" in cmdline.lower() or "next" in cmdline.lower():
                        rss = (proc.info.get("memory_info") or type("", (), {"rss": 0})).rss
                        procs.append(f"PID {proc.info['pid']}: {cmdline[:60]} ({rss // (1024**2)}MB)")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            result = f"Titan processes ({len(procs)}):\n" + "\n".join(procs) if procs else "No Titan processes found"
            return {"success": True, "result": result[:500], "enrichment_data": {},
                    "error": None}
        except ImportError:
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": "psutil not installed"}

    def status(self) -> str:
        """Always available (local system)."""
        return "available"
