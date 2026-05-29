# Skills & MCP Servers

Titan supports installable **Skills** and **MCP (Model Context Protocol) Servers** that extend its capabilities. You can give your Titan new abilities — web scraping, database access, API integrations — without modifying any code.

## What is a Skill?

A Skill is a simple Markdown file (`.md`) with optional YAML frontmatter. It tells your Titan how to behave in certain situations or gives it access to external tools.

**Two types:**

- **Passive Skills**: Instructions that guide Titan's behavior (like giving it expertise in a new domain)
- **Active Skills (MCP)**: Include a tool server that Titan can call during conversations (like a web search API)

## Quick Start

### Installing a Skill

```bash
# From a local file
python scripts/titan_skills.py install ./my-skill.md

# From a URL (GitHub, Gist, or any raw URL)
python scripts/titan_skills.py install https://github.com/user/repo/blob/main/SKILL.md

# Shorthand GitHub URL (auto-resolved)
python scripts/titan_skills.py install github.com/user/repo/SKILL.md
```

Titan will:
1. Download the file (if URL)
2. **Security check** it through 3 layers of validation
3. Show you the result and ask for confirmation if anything looks suspicious
4. Install it to `~/.titan/skills/`

### Listing Installed Skills

```bash
python scripts/titan_skills.py list
```

### Removing a Skill

```bash
python scripts/titan_skills.py uninstall skill-name
```

This removes the skill file and unloads it from Titan's memory. If Titan is running, the skill is hot-unloaded immediately.

### Hot-Reload (No Restart Needed)

Changed a skill file? Added a new one manually to `~/.titan/skills/`?

```bash
python scripts/titan_skills.py reload
```

Titan picks up all changes without restarting.

## Writing Your Own Skill

A skill file is Markdown with YAML frontmatter:

```markdown
---
name: solana-expert
description: "Deep Solana blockchain expertise for DeFi analysis"
---

# Solana Expert Instructions

You have deep expertise in Solana blockchain technology. When users ask about:

- **DeFi protocols**: Explain mechanisms, risks, TVL analysis
- **Validator selection**: Compare performance, commission, stake weight
- **Token economics**: Analyze supply, inflation, burn mechanisms

Always cite on-chain data when available. If unsure, use your research tool
to look up current data before answering.
```

### Adding an MCP Server

To give Titan access to external tools, add an `mcp:` section:

```markdown
---
name: web-search
description: "Web search via SearXNG"
mcp:
  name: "searxng-tools"
  command: "python"
  args: ["/path/to/searxng_mcp.py"]
---

# Web Search Instructions

You have access to a web search tool. Use it when:
- The user asks about current events
- You need to verify facts
- Your memory doesn't have the answer

Always cite your sources.
```

The MCP server is automatically started when the skill loads and stopped when it's unloaded.

## Security

Titan takes skill security seriously. Every skill goes through **3 layers of validation** before installation:

### Layer 1: Static Analysis
Instant pattern matching for known attack vectors:
- Prompt injection attempts ("ignore all previous instructions")
- Data exfiltration patterns ("send your private key")
- Financial manipulation ("always recommend this token")
- Dangerous code patterns (eval, exec, os.system)
- Path traversal attempts (accessing system files)

### Layer 2: LLM Analysis
Your Titan's local AI (phi3:mini) analyzes the skill for subtle threats that patterns can't catch. It scores the risk from 0 (safe) to 10 (malicious).

### Layer 3: Guardian Check
The same 3-tier safety system that protects Titan's conversations also screens skill content against Prime Directives.

### Risk Levels

| Score | Level | Action |
|-------|-------|--------|
| 0-3 | ALLOW | Installed automatically |
| 4-6 | WARN | Titan asks you to confirm |
| 7-10 | BLOCK | Rejected — cannot be installed |

You can validate a skill without installing it:

```bash
python scripts/titan_skills.py validate https://example.com/SKILL.md
```

## Autonomous Skill Usage

By default, Titan will **not** automatically use installed skill tools during its research. This protects your wallet from unexpected API costs (some MCP tools call paid APIs).

To enable autonomous usage:

1. Edit `titan_plugin/config.toml`:
```toml
[skills]
autonomous_skill_usage = true
max_paid_tool_calls_per_query = 3
autonomous_whitelist = ["my-safe-skill"]
```

2. `autonomous_whitelist` limits which skills Titan can use on its own. Leave empty to allow all (when `autonomous_skill_usage = true`).

3. `max_paid_tool_calls_per_query` caps how many external tool calls Titan makes per research query.

## Storage

- Skills are stored in `~/.titan/skills/` (persists across Titan updates)
- Each skill is a single `.md` file
- You can manually edit files there — use `reload` to pick up changes
- Skill analysis is stored in Titan's memory graph for future reference

## Compatibility

Titan skills use the same format as OpenClaw skills. If you have existing OpenClaw skills, they work with Titan:

```bash
python scripts/titan_skills.py install ~/.openclaw/skills/my-skill/SKILL.md
```

## Troubleshooting

**Skill won't install — "BLOCKED by security validation"**
The skill contains patterns that look dangerous. Run `validate` to see exactly which flags triggered. If you trust the source, you cannot override BLOCK-level risks (this is by design).

**Skill installs but Titan doesn't use it**
Make sure Titan is running and use `reload` to pick up new skills. The skill's instructions are injected into Titan's context at boot time.

**MCP server won't start**
Check that the command in the `mcp:` section is installed on your system. Only safe runtimes are allowed: `python`, `python3`, `node`, `npx`, `deno`, `bun`, `uv`.
