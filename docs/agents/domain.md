# Domain docs

**Layout:** Single-context.

- `CONTEXT.md` — project-wide domain language, glossary, and architecture overview
- `docs/adr/` — architecture decision records (not yet created)

## Skills (`app/skills/`)

Specialized capabilities packaged as "skills" that the agent loads on-demand at runtime, following the [LangChain skills pattern](https://docs.langchain.com/oss/python/langchain/multi-agent/skills).

### Key concepts

- **Skill** — a dataclass with `prompt` (the specialization prompt), `description` (short one-liner for the system prompt listing), and `references` (optional file paths/resources).
- **SKILL_REGISTRY** — module-level `dict[str, Skill]` mapping skill names to their definitions. Populated at import time or during startup.
- **load_skill** — a `@tool`-decorated function added to the agent's tool list. The agent calls it with a skill name and receives the skill's prompt plus any reference context.

### Architecture

Skills follow progressive disclosure: the base system prompt only tells the agent `load_skill` exists. When the agent encounters a question requiring deep domain knowledge (e.g., trail difficulty analysis, weather interpretation, safety assessment), it calls `load_skill` to load the relevant specialization. This keeps the context window lean.

### Extending

To add a new skill, create a `Skill` and insert it into `SKILL_REGISTRY`:

```python
from app.skills import SKILL_REGISTRY, Skill

SKILL_REGISTRY["trail_analysis"] = Skill(
    prompt="You are a trail analysis expert...",
    references=["app/skills/prompts/trails.md"],
)
```

Future extensions (tracked in the GitHub issue):
- **Dynamic tool registration** — loading a skill could register additional tools
- **Hierarchical skills** — skills exposing sub-skills
- **Markdown skill files** — skill prompts stored in `.agents/skills/` with frontmatter
