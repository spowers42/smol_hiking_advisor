from __future__ import annotations

from pathlib import Path

from langchain.tools import tool

from app.skills import get_registry


def _strip_frontmatter(text: str) -> str:
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return text
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return text
    body = "\n".join(lines[end + 1 :])
    return body.strip()


@tool
def load_skill(skill_name: str) -> str:
    """Load a specialized hiking skill prompt and context.

    Call this when you need deep domain knowledge. Available skill
    names are returned in the error message if you request an unknown one.

    Returns the skill's instructions and context.
    """
    registry = get_registry()
    if not registry:
        return "No skills are registered yet."

    skill = registry.get(skill_name)
    if skill is None:
        known = sorted(registry)
        return f"Unknown skill '{skill_name}'. Available skills: {', '.join(known)}"

    skill_file = Path(skill.path) / "SKILL.md"
    if not skill_file.exists():
        return f"Skill '{skill_name}' exists but SKILL.md was not found at {skill_file}."

    text = skill_file.read_text(encoding="utf-8")
    body = _strip_frontmatter(text)

    parts = [body]
    for ref in skill.references:
        ref_path = Path(skill.path) / ref
        if ref_path.exists():
            parts.append(f"\nReference: {ref_path.read_text(encoding='utf-8')}")
        else:
            parts.append(f"\nReference: {ref}")
    return "\n".join(parts)
