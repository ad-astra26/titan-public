"""
skills/installer.py — Smart skill installer with URL handling and validation.

Handles installation from:
  - Local file path: titan install ./my-skill.md
  - Full URL: titan install https://github.com/user/repo/blob/main/SKILL.md
  - Raw GitHub URL: titan install https://raw.githubusercontent.com/...
  - Partial GitHub: titan install github.com/user/repo/SKILL.md
  - Gist URL: titan install https://gist.github.com/user/hash

Validates content through the 3-layer security pipeline before installing.
"""
import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx

from .registry import SkillRegistry, Skill
from .validator import SkillValidator, ValidationResult
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)

# GitHub URL patterns for smart resolution
_GITHUB_BLOB_RE = re.compile(
    r"(?:https?://)?github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)"
)
_GITHUB_TREE_RE = re.compile(
    r"(?:https?://)?github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+)"
)
_GITHUB_PARTIAL_RE = re.compile(
    r"(?:https?://)?github\.com/([^/]+)/([^/]+)/?(.*)"
)
_GIST_RE = re.compile(
    r"(?:https?://)?gist\.github\.com/([^/]+)/([a-f0-9]+)"
)


class InstallResult:
    """Result of a skill installation attempt."""

    def __init__(self):
        self.success: bool = False
        self.skill_name: str = ""
        self.skill_hash: str = ""
        self.file_path: str = ""
        self.source: str = ""  # "local", "url", "github"
        self.validation: Optional[ValidationResult] = None
        self.error: str = ""
        self.requires_confirmation: bool = False

    def __repr__(self):
        status = "OK" if self.success else "FAIL"
        return f"InstallResult({status}, name={self.skill_name}, source={self.source})"


class SkillInstaller:
    """
    Smart skill installer with URL resolution and security validation.

    Flow:
      1. Resolve source (local file, URL, GitHub)
      2. Download content if remote
      3. Validate content type (is this actually a skill file?)
      4. Run 3-layer security validation
      5. Install to registry if safe
      6. Optionally store as research memory
    """

    def __init__(self, registry: SkillRegistry, validator: SkillValidator,
                 memory=None):
        self.registry = registry
        self.validator = validator
        self.memory = memory  # TieredMemoryGraph for research memory
        self._http_timeout = 30.0

    async def install(self, source: str, force: bool = False) -> InstallResult:
        """
        Install a skill from a file path or URL.

        Args:
            source: Local file path or URL to a skill .md file.
            force: If True, skip user confirmation for WARN-level risks.

        Returns:
            InstallResult with success status and details.
        """
        result = InstallResult()
        result.source = source

        # Step 1: Resolve and fetch content
        content, resolved_source, error = await self._fetch_content(source)
        if error:
            result.error = error
            return result

        # Step 2: Basic content validation (is this a skill file?)
        if not self._looks_like_skill(content):
            result.error = (
                "Content does not look like a valid skill file. "
                "Expected Markdown with YAML frontmatter (---) or skill instructions."
            )
            return result

        # Step 3: Security validation
        validation = await self.validator.validate(content, filename=resolved_source)
        result.validation = validation
        result.skill_hash = validation.skill_hash

        if validation.risk_level == "BLOCK":
            result.error = f"BLOCKED by security validation: {validation.summary}"
            logger.warning("[Installer] Blocked %s: %s", source, validation.summary)
            return result

        if validation.risk_level == "WARN" and not force:
            result.requires_confirmation = True
            result.error = f"CAUTION: {validation.summary}. Use --force to override."
            return result

        # Step 4: Validate MCP section if present
        from .registry import _split_frontmatter
        import yaml

        frontmatter, body = _split_frontmatter(content)
        mcp_config = None
        if frontmatter:
            try:
                meta = yaml.safe_load(frontmatter)
                if isinstance(meta, dict) and "mcp" in meta:
                    mcp_config = meta["mcp"]
                    mcp_safe, mcp_issues = self.validator.validate_mcp_section(mcp_config)
                    if not mcp_safe:
                        result.error = f"MCP validation failed: {'; '.join(mcp_issues)}"
                        return result
            except yaml.YAMLError:
                pass

        # Step 5: Save to skills directory
        skill_name = self._extract_name(content, resolved_source)
        safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', skill_name) + ".md"
        dest_path = self.registry.skills_dir / safe_filename

        dest_path.write_text(content, encoding="utf-8")
        logger.info("[Installer] Saved to %s", dest_path)

        # Step 6: Load into registry
        skill = self.registry.load_skill(str(dest_path))
        if not skill:
            result.error = "File saved but failed to parse as skill."
            return result

        # Store safety metadata on the skill
        skill.safety_score = validation.risk_score

        result.success = True
        result.skill_name = skill.name
        result.file_path = str(dest_path)

        # Step 7: Store as research memory (skill analysis)
        await self._store_skill_memory(skill, validation, resolved_source)

        logger.info("[Installer] Installed: %s (risk=%.1f, hash=%s)",
                    skill.name, validation.risk_score, validation.skill_hash)
        return result

    async def uninstall(self, name: str) -> bool:
        """
        Uninstall a skill by name. Removes file and unloads from registry.

        Returns:
            True if successfully uninstalled.
        """
        return self.registry.remove_skill(name)

    # ── Content Fetching ──

    async def _fetch_content(self, source: str) -> Tuple[str, str, str]:
        """
        Resolve source and fetch content.

        Returns:
            (content, resolved_source_label, error_or_empty)
        """
        source = source.strip()

        # Local file?
        if os.path.exists(source):
            return self._read_local(source)

        # Looks like a URL?
        if self._looks_like_url(source):
            return await self._fetch_url(source)

        # Could be a relative path that doesn't exist
        return "", source, f"File not found and not a valid URL: {source}"

    def _read_local(self, path: str) -> Tuple[str, str, str]:
        """Read a local file."""
        try:
            content = Path(path).read_text(encoding="utf-8")
            return content, f"local:{path}", ""
        except Exception as e:
            return "", path, f"Failed to read file: {e}"

    async def _fetch_url(self, url: str) -> Tuple[str, str, str]:
        """Fetch content from a URL with smart GitHub resolution."""
        resolved_url = self._resolve_url(url)

        try:
            async with httpx.AsyncClient(
                timeout=self._http_timeout,
                follow_redirects=True,
                headers={"User-Agent": "Titan-Skill-Installer/1.0"},
            ) as client:
                resp = await client.get(resolved_url)

                if resp.status_code != 200:
                    return "", url, f"HTTP {resp.status_code} from {resolved_url}"

                content_type = resp.headers.get("content-type", "")

                # Reject binary content
                if any(t in content_type for t in ["image/", "application/octet", "application/zip"]):
                    return "", url, f"Not a text file (content-type: {content_type})"

                content = resp.text

                # Reject HTML pages (user pasted a GitHub page URL instead of raw)
                if content.strip().startswith("<!DOCTYPE") or content.strip().startswith("<html"):
                    # Try to auto-resolve to raw
                    raw_url = self._try_raw_conversion(url)
                    if raw_url and raw_url != resolved_url:
                        logger.debug("[Installer] Got HTML, trying raw URL: %s", raw_url)
                        resp2 = await client.get(raw_url)
                        if resp2.status_code == 200 and not resp2.text.strip().startswith("<"):
                            return resp2.text, f"url:{raw_url}", ""

                    return "", url, "URL returned HTML instead of raw Markdown. Use the raw file URL."

                return content, f"url:{resolved_url}", ""

        except httpx.TimeoutException:
            return "", url, f"Timeout fetching {resolved_url}"
        except Exception as e:
            return "", url, f"Fetch error: {e}"

    def _resolve_url(self, url: str) -> str:
        """
        Smart URL resolution: convert GitHub/Gist URLs to raw content URLs.
        """
        url = url.strip()

        # Ensure scheme
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        # GitHub blob → raw
        m = _GITHUB_BLOB_RE.match(url)
        if m:
            user, repo, branch, path = m.groups()
            return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"

        # GitHub tree → raw (same conversion)
        m = _GITHUB_TREE_RE.match(url)
        if m:
            user, repo, branch, path = m.groups()
            return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"

        # Gist → raw
        m = _GIST_RE.match(url)
        if m:
            user, gist_id = m.groups()
            return f"https://gist.githubusercontent.com/{user}/{gist_id}/raw"

        # Partial GitHub (github.com/user/repo/file.md) → try main branch raw
        m = _GITHUB_PARTIAL_RE.match(url)
        if m:
            user, repo, rest = m.groups()
            if rest and rest.endswith(".md"):
                return f"https://raw.githubusercontent.com/{user}/{repo}/main/{rest}"

        # Already a raw URL or other provider
        return url

    def _try_raw_conversion(self, url: str) -> Optional[str]:
        """Attempt to convert any GitHub URL to raw."""
        if "github.com" in url and "/blob/" in url:
            return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        return None

    # ── Content Validation ──

    @staticmethod
    def _looks_like_skill(content: str) -> bool:
        """
        Basic sanity check: does this content look like a skill file?
        """
        content = content.strip()
        if not content:
            return False

        # Too short to be useful
        if len(content) < 20:
            return False

        # Has YAML frontmatter → likely a skill
        if content.startswith("---"):
            return True

        # Has Markdown headers → could be skill instructions
        if re.search(r'^#+\s', content, re.MULTILINE):
            return True

        # Has meaningful text content (not just code or data)
        word_count = len(content.split())
        return word_count >= 10

    @staticmethod
    def _looks_like_url(source: str) -> bool:
        """Check if source looks like a URL."""
        source = source.strip()
        if source.startswith("http://") or source.startswith("https://"):
            return True
        # Partial URLs
        if source.startswith("github.com/") or source.startswith("gist.github.com/"):
            return True
        # Domain-like pattern
        if re.match(r'^[\w.-]+\.\w{2,}/', source):
            return True
        return False

    @staticmethod
    def _extract_name(content: str, source: str) -> str:
        """Extract skill name from frontmatter or source."""
        from .registry import _split_frontmatter
        import yaml

        frontmatter, _ = _split_frontmatter(content)
        if frontmatter:
            try:
                meta = yaml.safe_load(frontmatter)
                if isinstance(meta, dict) and meta.get("name"):
                    return meta["name"]
            except yaml.YAMLError:
                pass

        # Fall back to filename from source
        if "/" in source:
            basename = source.rstrip("/").rsplit("/", 1)[-1]
            if basename.endswith(".md"):
                return basename[:-3]

        return f"skill-{hashlib.sha256(content.encode()).hexdigest()[:8]}"

    # ── Research Memory ──

    async def _store_skill_memory(self, skill: Skill, validation: ValidationResult,
                                  source: str):
        """Store skill analysis as research memory for Titan's knowledge base."""
        if not self.memory:
            return

        try:
            summary = (
                f"Installed skill '{skill.name}': {skill.description}. "
                f"Source: {source}. Safety score: {validation.risk_score}/10. "
                f"Hash: {validation.skill_hash}. "
            )
            if skill.mcp:
                tools = skill.mcp.get("name", "unknown")
                summary += f"Provides MCP server '{tools}'. "

            summary += f"Capabilities: {skill.body[:200]}"

            await self.memory.add_to_mempool(
                user_prompt=f"[SKILL_INSTALL] Analyzed and installed skill: {skill.name}",
                agent_response=summary,
                user_identifier="titan_skill_installer",
            )
            self.memory.add_research_topic(f"skill:{skill.name}")
            logger.debug("[Installer] Skill memory stored for %s", skill.name)
        except Exception as e:
            swallow_warn('[Installer] Failed to store skill memory', e,
                         key="skills.installer.failed_to_store_skill_memory", throttle=100)
