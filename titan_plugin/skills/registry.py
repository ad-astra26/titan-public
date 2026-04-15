"""
skills/registry.py — Skill registry with hot-reload support.

Manages installed skills: load, unload, list, hot-reload.
Skills are stored as .md files in ~/.titan/skills/ with parsed metadata cached in-memory.
"""
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Default skills directory
DEFAULT_SKILLS_DIR = os.path.expanduser("~/.titan/skills")


@dataclass
class Skill:
    """Represents a loaded skill."""
    name: str
    description: str
    file_path: str
    content_hash: str
    body: str  # Markdown body (injected into agent context)
    mcp: Optional[dict] = None  # MCP config if present
    installed_at: float = 0.0
    safety_score: float = 0.0
    capabilities: str = ""
    tools_provided: list = field(default_factory=list)
    active: bool = True


class SkillRegistry:
    """
    In-memory registry of installed skills with hot-reload support.

    Skills are .md files with YAML frontmatter in ~/.titan/skills/.
    The registry parses and caches them, providing context injection
    and MCP spawn information to the agent.
    """

    def __init__(self, skills_dir: str = DEFAULT_SKILLS_DIR):
        self._skills_dir = Path(skills_dir)
        self._skills: Dict[str, Skill] = {}
        self._ensure_dir()

    def _ensure_dir(self):
        """Create skills directory if it doesn't exist."""
        self._skills_dir.mkdir(parents=True, exist_ok=True)

    @property
    def skills_dir(self) -> Path:
        return self._skills_dir

    # ── Load / Unload ──

    def load_all(self) -> int:
        """
        Load all .md files from the skills directory.
        Returns the number of skills loaded.
        """
        count = 0
        for md_file in sorted(self._skills_dir.glob("*.md")):
            try:
                skill = self._parse_skill_file(md_file)
                if skill:
                    self._skills[skill.name] = skill
                    count += 1
                    logger.info("[SkillRegistry] Loaded: %s (%s)", skill.name, md_file.name)
            except Exception as e:
                logger.warning("[SkillRegistry] Failed to load %s: %s", md_file.name, e)
        return count

    def load_skill(self, file_path: str) -> Optional[Skill]:
        """
        Load a single skill from a file path.
        If a skill with the same name exists, it is replaced (hot-reload).

        Returns:
            The loaded Skill, or None if parsing failed.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning("[SkillRegistry] File not found: %s", file_path)
            return None

        skill = self._parse_skill_file(path)
        if skill:
            old = self._skills.get(skill.name)
            if old:
                logger.info("[SkillRegistry] Replacing skill: %s (hot-reload)", skill.name)
            self._skills[skill.name] = skill
            return skill
        return None

    def unload_skill(self, name: str) -> bool:
        """
        Unload a skill by name. Does NOT delete the file.

        Returns:
            True if the skill was found and unloaded.
        """
        if name in self._skills:
            self._skills[name].active = False
            del self._skills[name]
            logger.info("[SkillRegistry] Unloaded: %s", name)
            return True
        logger.warning("[SkillRegistry] Skill not found: %s", name)
        return False

    def remove_skill(self, name: str) -> bool:
        """
        Unload and delete the skill file.

        Returns:
            True if the skill was found, unloaded, and file deleted.
        """
        skill = self._skills.get(name)
        if not skill:
            logger.warning("[SkillRegistry] Skill not found: %s", name)
            return False

        # Delete file
        file_path = Path(skill.file_path)
        if file_path.exists():
            file_path.unlink()
            logger.info("[SkillRegistry] Deleted file: %s", file_path)

        del self._skills[name]
        return True

    # ── Query ──

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a loaded skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> List[Skill]:
        """List all loaded skills."""
        return list(self._skills.values())

    def get_active_skills(self) -> List[Skill]:
        """List only active skills."""
        return [s for s in self._skills.values() if s.active]

    def get_combined_context(self) -> str:
        """
        Build combined context string from all active skills.
        This is injected into agent.additional_context.
        """
        active = self.get_active_skills()
        if not active:
            return ""

        parts = ["### Installed Skills"]
        for skill in active:
            parts.append(f"\n#### {skill.name}")
            if skill.description:
                parts.append(f"*{skill.description}*")
            parts.append(skill.body)
        return "\n".join(parts)

    def get_mcp_skills(self) -> List[Skill]:
        """List skills that have MCP server configurations."""
        return [s for s in self._skills.values() if s.active and s.mcp]

    # ── Hot Reload ──

    def reload_all(self) -> dict:
        """
        Reload all skills from disk. Detects added, removed, and changed files.

        Returns:
            Dict with counts: {"added": n, "removed": n, "updated": n, "unchanged": n}
        """
        result = {"added": 0, "removed": 0, "updated": 0, "unchanged": 0}

        # Scan disk
        disk_skills = {}
        for md_file in self._skills_dir.glob("*.md"):
            try:
                skill = self._parse_skill_file(md_file)
                if skill:
                    disk_skills[skill.name] = skill
            except Exception as e:
                logger.warning("[SkillRegistry] Failed to parse %s during reload: %s", md_file.name, e)

        # Detect removals (in memory but not on disk)
        current_names = set(self._skills.keys())
        disk_names = set(disk_skills.keys())
        for removed in current_names - disk_names:
            del self._skills[removed]
            result["removed"] += 1
            logger.info("[SkillRegistry] Removed (file deleted): %s", removed)

        # Detect additions and updates
        for name, skill in disk_skills.items():
            old = self._skills.get(name)
            if old is None:
                self._skills[name] = skill
                result["added"] += 1
                logger.info("[SkillRegistry] Added: %s", name)
            elif old.content_hash != skill.content_hash:
                self._skills[name] = skill
                result["updated"] += 1
                logger.info("[SkillRegistry] Updated: %s", name)
            else:
                result["unchanged"] += 1

        logger.info("[SkillRegistry] Reload complete: %s", result)
        return result

    # ── Parsing ──

    @staticmethod
    def _parse_skill_file(path: Path) -> Optional[Skill]:
        """
        Parse a skill .md file with YAML frontmatter.

        Expected format:
            ---
            name: skill-name
            description: "What it does"
            mcp:                    # Optional
              name: "server-name"
              command: "python"
              args: ["path/to/server.py"]
            ---
            # Markdown body (instructions for the LLM)
            ...
        """
        content = path.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Split frontmatter and body
        frontmatter, body = _split_frontmatter(content)
        if frontmatter is None:
            # No frontmatter — treat entire content as body, use filename as name
            name = path.stem
            return Skill(
                name=name,
                description="",
                file_path=str(path),
                content_hash=content_hash,
                body=content.strip(),
                installed_at=path.stat().st_mtime,
            )

        try:
            meta = yaml.safe_load(frontmatter)
        except yaml.YAMLError as e:
            logger.warning("[SkillRegistry] Invalid YAML in %s: %s", path.name, e)
            return None

        if not isinstance(meta, dict):
            logger.warning("[SkillRegistry] Frontmatter is not a dict in %s", path.name)
            return None

        name = meta.get("name", path.stem)
        mcp_config = meta.get("mcp")

        return Skill(
            name=name,
            description=meta.get("description", ""),
            file_path=str(path),
            content_hash=content_hash,
            body=body.strip(),
            mcp=mcp_config,
            installed_at=path.stat().st_mtime,
        )


def _split_frontmatter(content: str) -> tuple:
    """
    Split YAML frontmatter from Markdown body.

    Returns:
        (frontmatter_str, body_str) or (None, None) if no frontmatter.
    """
    content = content.strip()
    if not content.startswith("---"):
        return None, None

    # Find closing ---
    end_idx = content.find("---", 3)
    if end_idx == -1:
        return None, None

    frontmatter = content[3:end_idx].strip()
    body = content[end_idx + 3:].strip()
    return frontmatter, body
