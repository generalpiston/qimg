# OpenClaw Skill Configuration

This folder contains OpenClaw skill templates for this repository.

## Skill File Format

OpenClaw skills use a `SKILL.md` file with YAML frontmatter and markdown body.

Minimal template:

```md
---
name: "qimg-retrieval"
description: "Query and inspect indexed images with qimg MCP tools."
allowed-tools:
  - "mcp__qimg__qimg_status"
  - "mcp__qimg__qimg_deep_search"
  - "mcp__qimg__qimg_get"
---

# Qimg Retrieval

Use this skill when you need to search and inspect image assets from qimg.
```

## Loading and Scope

OpenClaw supports global and project-level skills. For project skills, place files in:

```text
.openclaw/skills/
```

Skill precedence is project-first over global when names conflict.

## Skill Management Commands

Common commands:

```bash
openclaw skills discover
openclaw skills list
openclaw skills view <skill-name>
openclaw skills enable <skill-name>
openclaw skills disable <skill-name>
```

## References

- https://docs.openclaw.ai/skills/format
- https://docs.openclaw.ai/skills/loading-and-precedence
- https://docs.openclaw.ai/getting-started/skills
