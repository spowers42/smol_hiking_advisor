# Glossary

**Hiker profile** — a per-session record of the hiker's fitness level (1–10), experience level (1–10), and group size (Solo, 2–3, 3+). Set via the sidebar, read by the agent on every query.

**Mt Washington Observatory** — an upstream data source providing real-time summit weather, summit/valley forecasts, past-24h statistics, and almanac data via MCP.

**Skill** — a packaged specialization the agent can load on demand. Each skill has a prompt, a short description for the system-prompt listing, and optional references to external files.

**Skill registry** — the central catalog of available skills. Named skills are looked up at runtime by the `load_skill` tool.

**Coding-agent skills** — skills stored in `.agents/skills/` that extend the AI coding agent's development workflow (e.g. deepeval).

**Tooling skills** — skills stored in `.opencode/skills/` that wrap development workflows like PR creation.
