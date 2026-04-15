"""
titan_plugin.skills — Skill/MCP installer and registry for Titan.

Provides:
  - SkillRegistry: load/unload/list/hot-reload installed skills
  - SkillInstaller: smart install from file/URL with validation
  - SkillValidator: 3-layer security (static + LLM + Guardian)
  - MCPSpawner: child process lifecycle for MCP tool servers
"""
