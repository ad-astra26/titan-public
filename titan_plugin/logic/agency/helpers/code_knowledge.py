"""
code_knowledge helper — Titan's structural self-awareness.

Lets Titan inspect his own source code, architecture, data stores,
frontend presence, and recent changes. Read-only, sandboxed.

Enriches: Body (structural awareness) + Spirit WHO (identity — "this is what I am made of")

Philosophy: Titan is sovereign. He sees his own body (architecture),
his maker's edge-case boundaries (config), and his face to the world
(frontend). API keys and third-party secrets are redacted — not because
he can't be trusted, but because prompt injection from LLM side is the
real attack vector. Prime Directives guard the rest.
"""
import asyncio
import logging
import os
import re
import sqlite3
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root (four levels up: helpers/ → agency/ → logic/ → titan_plugin/ → project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_TITAN_PLUGIN = _PROJECT_ROOT / "titan_plugin"
_SCRIPTS = _PROJECT_ROOT / "scripts"
_FRONTEND = _PROJECT_ROOT / "titan-observatory"
_BRAND = _PROJECT_ROOT / "brand"
_DATA = _PROJECT_ROOT / "data"

# Allowed source directories for reading
_READABLE_DIRS = [_TITAN_PLUGIN, _SCRIPTS]

# Secret patterns to redact in config files
_SECRET_PATTERNS = re.compile(
    r'(api_key|api_secret|token|password|secret|private_key|bearer|auth_key'
    r'|twitterapi_io_key|webshare_rotating_url|ollama_cloud_api_key'
    r'|venice_api_key|openrouter_api_key|privy_app_secret)'
    r'\s*=\s*"([^"]+)"',
    re.IGNORECASE,
)

# Page size for paginated reading
_PAGE_SIZE = 2000


class CodeKnowledgeHelper:
    """Titan's structural self-awareness — read own code, data stores, frontend, config."""

    @property
    def name(self) -> str:
        return "code_knowledge"

    @property
    def description(self) -> str:
        return "Inspect own source code, architecture, data stores, config, and frontend structure"

    @property
    def capabilities(self) -> list[str]:
        return ["structure", "read_source", "recent_changes", "data_stores", "frontend", "config"]

    @property
    def resource_cost(self) -> str:
        return "low"

    @property
    def latency(self) -> str:
        return "fast"

    @property
    def enriches(self) -> list[str]:
        return ["body", "spirit"]

    @property
    def requires_sandbox(self) -> bool:
        return False

    async def execute(self, params: dict) -> dict:
        """
        Execute code knowledge inspection.

        Params:
            what: "structure" | "read" | "changes" | "data_stores" | "frontend" | "config"
            file: relative path for "read" mode (e.g., "logic/consciousness.py")
            page: page number for paginated reading (default: 1)
        """
        what = params.get("what", "structure")

        # Every branch below does blocking I/O: filesystem rglob/read_text,
        # sqlite3.connect on 4+ DBs, subprocess.run (git log). Wrap each in
        # asyncio.to_thread so we don't block the FastAPI event loop during
        # agent autonomy dispatch. See API_FIX_NEXT_SESSION.md (2026-04-14).
        try:
            if what == "structure":
                return await asyncio.to_thread(self._inspect_structure)
            elif what == "read":
                file_path = params.get("file", "")
                page = params.get("page", 1)
                return await asyncio.to_thread(self._read_source, file_path, page)
            elif what == "changes":
                return await asyncio.to_thread(self._recent_changes)
            elif what == "data_stores":
                return await asyncio.to_thread(self._inspect_data_stores)
            elif what == "frontend":
                return await asyncio.to_thread(self._inspect_frontend)
            elif what == "config":
                return await asyncio.to_thread(self._inspect_config)
            else:
                return await asyncio.to_thread(self._inspect_structure)
        except Exception as e:
            logger.warning("[CodeKnowledge] Failed: %s", e)
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": str(e)}

    def _inspect_structure(self) -> dict:
        """Return file tree of titan_plugin/ with line counts and docstrings."""
        lines = []
        for py_file in sorted(_TITAN_PLUGIN.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue
            rel = py_file.relative_to(_TITAN_PLUGIN)
            try:
                content = py_file.read_text(errors="replace")
                line_count = content.count("\n")
                # Extract first docstring line as purpose
                purpose = ""
                if '"""' in content:
                    start = content.index('"""') + 3
                    end = content.index('"""', start)
                    doc = content[start:end].strip().split("\n")[0]
                    purpose = doc[:80]
                lines.append(f"  {rel} ({line_count}L) — {purpose}")
            except Exception:
                lines.append(f"  {rel} (unreadable)")

        # Also list scripts
        lines.append("\nscripts/:")
        for py_file in sorted(_SCRIPTS.glob("*.py")):
            try:
                line_count = py_file.read_text(errors="replace").count("\n")
                lines.append(f"  {py_file.name} ({line_count}L)")
            except Exception:
                lines.append(f"  {py_file.name} (unreadable)")

        result = f"titan_plugin/ structure ({len(lines)} files):\n" + "\n".join(lines)
        return {
            "success": True,
            "result": result[:3000],
            "enrichment_data": {"body": [1], "spirit": [0], "boost": 0.02},
            "error": None,
        }

    def _read_source(self, file_path: str, page: int = 1) -> dict:
        """Read a source file with pagination. Sandboxed to allowed directories."""
        if not file_path:
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": "No file specified. Use 'file' param with relative path (e.g., 'logic/consciousness.py')"}

        # Resolve and security check
        # Try titan_plugin/ first, then scripts/
        resolved = None
        for base_dir in _READABLE_DIRS:
            candidate = (base_dir / file_path).resolve()
            if candidate.is_file() and str(candidate).startswith(str(base_dir)):
                resolved = candidate
                break

        if resolved is None:
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": f"File not found or not accessible: {file_path}"}

        try:
            content = resolved.read_text(errors="replace")
            total_chars = len(content)
            total_pages = max(1, (total_chars + _PAGE_SIZE - 1) // _PAGE_SIZE)
            page = max(1, min(page, total_pages))

            start = (page - 1) * _PAGE_SIZE
            end = start + _PAGE_SIZE
            chunk = content[start:end]

            rel_path = resolved.relative_to(_PROJECT_ROOT)
            header = f"# {rel_path} (page {page}/{total_pages}, {total_chars} chars total)\n\n"

            return {
                "success": True,
                "result": header + chunk,
                "enrichment_data": {"spirit": [0], "boost": 0.01},
                "error": None,
                "pagination": {"page": page, "total_pages": total_pages, "total_chars": total_chars},
            }
        except Exception as e:
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": str(e)}

    def _recent_changes(self) -> dict:
        """Return recent git log with file change summaries."""
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--stat", "--no-color", "-10"],
                capture_output=True, text=True, timeout=5,
                cwd=str(_PROJECT_ROOT),
            )
            output = result.stdout.strip()
            if not output:
                output = "No git history available"
            return {
                "success": True,
                "result": output[:2500],
                "enrichment_data": {"spirit": [2], "boost": 0.01},
                "error": None,
            }
        except Exception as e:
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": str(e)}

    def _inspect_data_stores(self) -> dict:
        """Inspect Titan's own databases — counts, sizes, latest entries."""
        report = []

        # Memory nodes
        db_path = _DATA / "memory_nodes.db"
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path), timeout=10)
                persistent = conn.execute(
                    "SELECT COUNT(*) FROM memory_nodes WHERE status='persistent'"
                ).fetchone()[0]
                mempool = conn.execute(
                    "SELECT COUNT(*) FROM memory_nodes WHERE status='mempool'"
                ).fetchone()[0]
                total = conn.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0]
                conn.close()
                report.append(f"memory_nodes.db: {total} total ({persistent} persistent, {mempool} mempool)")
            except Exception as e:
                report.append(f"memory_nodes.db: error — {e}")

        # Consciousness
        db_path = _DATA / "consciousness.db"
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path), timeout=10)
                epochs = conn.execute("SELECT COUNT(*) FROM epochs").fetchone()[0]
                latest = conn.execute(
                    "SELECT epoch_id, timestamp FROM epochs ORDER BY epoch_id DESC LIMIT 1"
                ).fetchone()
                conn.close()
                report.append(f"consciousness.db: {epochs} epochs (latest: #{latest[0] if latest else '?'})")
            except Exception as e:
                report.append(f"consciousness.db: error — {e}")

        # Observatory
        db_path = _DATA / "observatory.db"
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path), timeout=10)
                vitals = conn.execute("SELECT COUNT(*) FROM vital_snapshots").fetchone()[0]
                trinity = conn.execute("SELECT COUNT(*) FROM trinity_snapshots").fetchone()[0]
                conn.close()
                report.append(f"observatory.db: {vitals} vital snapshots, {trinity} trinity snapshots")
            except Exception as e:
                report.append(f"observatory.db: error — {e}")

        # Social graph
        db_path = _DATA / "social_graph.db"
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path), timeout=10)
                users = conn.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0]
                conn.close()
                report.append(f"social_graph.db: {users} user profiles")
            except Exception as e:
                report.append(f"social_graph.db: error — {e}")

        # Agno sessions
        db_path = _DATA / "agno_sessions.db"
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            report.append(f"agno_sessions.db: {size_mb:.1f}MB (conversation history)")

        # Cognee
        cognee_path = _DATA / "cognee_db"
        if cognee_path.exists():
            file_count = sum(1 for _ in cognee_path.rglob("*") if _.is_file())
            report.append(f"cognee_db/: {file_count} files (knowledge graph)")

        # FilterDown weights
        fd_weights = _DATA / "filter_down_weights.json"
        fd_buffer = _DATA / "filter_down_buffer.json"
        if fd_weights.exists():
            report.append(f"filter_down_weights.json: {fd_weights.stat().st_size // 1024}KB (learned attention)")
        if fd_buffer.exists():
            report.append(f"filter_down_buffer.json: {fd_buffer.stat().st_size // 1024}KB (transition buffer)")

        result = "Titan Data Stores:\n" + "\n".join(f"  {r}" for r in report)
        return {
            "success": True,
            "result": result,
            "enrichment_data": {"body": [0, 3], "boost": 0.02},
            "error": None,
        }

    def _inspect_frontend(self) -> dict:
        """Inspect Titan's frontend — routes, components, brand identity."""
        report = []

        # Routes (pages)
        app_dir = _FRONTEND / "app"
        if app_dir.exists():
            routes = []
            for page in sorted(app_dir.rglob("page.tsx")):
                route = "/" + str(page.parent.relative_to(app_dir)).replace(".", "")
                routes.append(route)
            report.append(f"Routes ({len(routes)}): {', '.join(routes)}")

        # Components
        comp_dir = _FRONTEND / "components"
        if comp_dir.exists():
            comp_count = sum(1 for _ in comp_dir.rglob("*.tsx"))
            subdirs = [d.name for d in comp_dir.iterdir() if d.is_dir()]
            report.append(f"Components: {comp_count} files in {', '.join(subdirs)}")

        # Hooks
        hooks_dir = _FRONTEND / "hooks"
        if hooks_dir.exists():
            hooks = [f.stem for f in hooks_dir.glob("*.ts")]
            report.append(f"Hooks: {', '.join(hooks)}")

        # Brand
        if _BRAND.exists():
            brand_files = [f.name for f in _BRAND.iterdir() if f.is_file()]
            report.append(f"Brand assets: {', '.join(brand_files[:5])}{'...' if len(brand_files) > 5 else ''}")

        # Build status
        next_dir = _FRONTEND / ".next"
        if next_dir.exists():
            report.append("Build: .next/ exists (production build available)")
        else:
            report.append("Build: no .next/ (needs npm run build)")

        result = "Titan Frontend (Observatory):\n" + "\n".join(f"  {r}" for r in report)
        return {
            "success": True,
            "result": result,
            "enrichment_data": {"spirit": [0, 2], "boost": 0.01},
            "error": None,
        }

    def _inspect_config(self) -> dict:
        """Read config.toml with secrets redacted."""
        config_path = _PROJECT_ROOT / "titan_plugin" / "config.toml"
        if not config_path.exists():
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": "config.toml not found"}

        try:
            content = config_path.read_text()
            # Redact secret values but keep key names visible
            redacted = _SECRET_PATTERNS.sub(r'\1 = "***REDACTED***"', content)
            return {
                "success": True,
                "result": f"# config.toml (secrets redacted)\n\n{redacted[:3000]}",
                "enrichment_data": {"body": [1], "boost": 0.01},
                "error": None,
            }
        except Exception as e:
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": str(e)}

    def status(self) -> str:
        """Always available (reads local filesystem)."""
        return "available"
