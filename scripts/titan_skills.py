#!/usr/bin/env python3
"""
titan_skills.py — CLI for managing Titan skills and MCP servers.

Usage:
    python scripts/titan_skills.py install <path-or-url>   # Install a skill
    python scripts/titan_skills.py install <url> --force    # Install, skip WARN confirmation
    python scripts/titan_skills.py uninstall <name>         # Uninstall by name
    python scripts/titan_skills.py list                     # List installed skills
    python scripts/titan_skills.py reload                   # Hot-reload all skills
    python scripts/titan_skills.py info <name>              # Show skill details
    python scripts/titan_skills.py validate <path-or-url>   # Validate without installing
"""
import argparse
import asyncio
import logging
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config() -> dict:
    try:
        from titan_plugin.config_loader import load_titan_config
        return load_titan_config()
    except Exception:
        return {}


def _build_components(config: dict):
    """Build installer components without booting full TitanPlugin."""
    from titan_plugin.skills.registry import SkillRegistry
    from titan_plugin.skills.validator import SkillValidator
    from titan_plugin.skills.installer import SkillInstaller

    skills_cfg = config.get("skills", {})
    skills_dir = os.path.expanduser(skills_cfg.get("skills_dir", "~/.titan/skills"))

    inference_cfg = config.get("inference", {})
    stealth_cfg = config.get("stealth_sage", {})

    registry = SkillRegistry(skills_dir=skills_dir)
    validator = SkillValidator(
        ollama_host=stealth_cfg.get("ollama_host", "http://localhost:11434"),
        ollama_model=stealth_cfg.get("ollama_model", "phi3:mini"),
    )
    installer = SkillInstaller(registry, validator)

    return registry, validator, installer


async def cmd_install(args, config):
    """Install a skill from file or URL."""
    registry, validator, installer = _build_components(config)

    print(f"\n  Installing skill from: {args.source}")
    print("  Validating security...")

    result = await installer.install(args.source, force=args.force)

    if result.requires_confirmation:
        print(f"\n  WARNING: {result.error}")
        print(f"  Validation: {result.validation.summary}")
        if result.validation.static_flags:
            print(f"  Flags: {', '.join(result.validation.static_flags)}")

        answer = input("\n  Install anyway? (yes/no): ").strip().lower()
        if answer in ("yes", "y"):
            result = await installer.install(args.source, force=True)
        else:
            print("  Installation cancelled.")
            return

    if result.success:
        print(f"\n  Installed: {result.skill_name}")
        print(f"  File: {result.file_path}")
        print(f"  Hash: {result.skill_hash}")
        print(f"  Safety: {result.validation.risk_level} (score={result.validation.risk_score})")
        skill = registry.get_skill(result.skill_name)
        if skill and skill.mcp:
            print(f"  MCP server: {skill.mcp.get('name', 'unnamed')}")
        print()
    else:
        print(f"\n  FAILED: {result.error}\n")
        sys.exit(1)


async def cmd_uninstall(args, config):
    """Uninstall a skill by name."""
    registry, _, installer = _build_components(config)
    registry.load_all()

    skill = registry.get_skill(args.name)
    if not skill:
        print(f"\n  Skill not found: {args.name}")
        print(f"  Installed skills: {[s.name for s in registry.list_skills()]}\n")
        sys.exit(1)

    success = await installer.uninstall(args.name)
    if success:
        print(f"\n  Uninstalled: {args.name}")
        print(f"  File deleted.\n")
    else:
        print(f"\n  Failed to uninstall: {args.name}\n")
        sys.exit(1)


async def cmd_list(args, config):
    """List installed skills."""
    registry, _, _ = _build_components(config)
    count = registry.load_all()

    skills = registry.list_skills()
    if not skills:
        print("\n  No skills installed.")
        print(f"  Skills directory: {registry.skills_dir}\n")
        return

    print(f"\n  Installed Skills ({count}):")
    print(f"  {'─' * 60}")
    for s in skills:
        mcp_label = f" [MCP: {s.mcp.get('name', '?')}]" if s.mcp else ""
        safety = f" (risk={s.safety_score:.1f})" if s.safety_score > 0 else ""
        print(f"  {s.name:<30} {s.description[:40]}{mcp_label}{safety}")
    print(f"  {'─' * 60}")
    print(f"  Directory: {registry.skills_dir}\n")


async def cmd_reload(args, config):
    """Hot-reload all skills from disk."""
    registry, _, _ = _build_components(config)
    registry.load_all()  # Load current state first
    result = registry.reload_all()

    print(f"\n  Reload complete:")
    print(f"    Added:     {result['added']}")
    print(f"    Updated:   {result['updated']}")
    print(f"    Removed:   {result['removed']}")
    print(f"    Unchanged: {result['unchanged']}\n")


async def cmd_info(args, config):
    """Show detailed info about an installed skill."""
    registry, _, _ = _build_components(config)
    registry.load_all()

    skill = registry.get_skill(args.name)
    if not skill:
        print(f"\n  Skill not found: {args.name}\n")
        sys.exit(1)

    print(f"\n  Skill: {skill.name}")
    print(f"  Description: {skill.description}")
    print(f"  File: {skill.file_path}")
    print(f"  Hash: {skill.content_hash}")
    print(f"  Safety Score: {skill.safety_score}")
    print(f"  Active: {skill.active}")
    if skill.mcp:
        print(f"  MCP Server: {skill.mcp.get('name', 'unnamed')}")
        print(f"  MCP Command: {skill.mcp.get('command', '')} {' '.join(str(a) for a in skill.mcp.get('args', []))}")
    print(f"\n  Body ({len(skill.body)} chars):")
    print(f"  {'─' * 60}")
    for line in skill.body[:500].split("\n"):
        print(f"  {line}")
    if len(skill.body) > 500:
        print(f"  ... ({len(skill.body) - 500} more chars)")
    print(f"  {'─' * 60}\n")


async def cmd_validate(args, config):
    """Validate a skill without installing it."""
    _, validator, installer = _build_components(config)

    print(f"\n  Validating: {args.source}")

    # Fetch content
    content, source_label, error = await installer._fetch_content(args.source)
    if error:
        print(f"  ERROR: {error}\n")
        sys.exit(1)

    # Validate
    result = await validator.validate(content, filename=source_label)

    print(f"\n  Risk Level: {result.risk_level}")
    print(f"  Risk Score: {result.risk_score}")
    print(f"  Hash: {result.skill_hash}")
    print(f"  Summary: {result.summary}")
    if result.static_flags:
        print(f"  Static Flags: {', '.join(result.static_flags)}")
    if result.llm_analysis:
        print(f"  LLM Analysis: {result.llm_analysis}")
    if result.guardian_safe is not None:
        print(f"  Guardian: {'SAFE' if result.guardian_safe else 'BLOCKED'}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Titan Skill/MCP Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # install
    p_install = subparsers.add_parser("install", help="Install a skill from file or URL")
    p_install.add_argument("source", help="File path or URL to skill .md file")
    p_install.add_argument("--force", action="store_true", help="Skip WARN confirmation")

    # uninstall
    p_uninstall = subparsers.add_parser("uninstall", help="Uninstall a skill by name")
    p_uninstall.add_argument("name", help="Skill name")

    # list
    subparsers.add_parser("list", help="List installed skills")

    # reload
    subparsers.add_parser("reload", help="Hot-reload all skills from disk")

    # info
    p_info = subparsers.add_parser("info", help="Show skill details")
    p_info.add_argument("name", help="Skill name")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate without installing")
    p_validate.add_argument("source", help="File path or URL to validate")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging()
    config = _load_config()

    cmd_map = {
        "install": cmd_install,
        "uninstall": cmd_uninstall,
        "list": cmd_list,
        "reload": cmd_reload,
        "info": cmd_info,
        "validate": cmd_validate,
    }

    asyncio.run(cmd_map[args.command](args, config))


if __name__ == "__main__":
    main()
