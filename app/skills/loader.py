from langchain.tools import tool

from app.skills import SKILL_REGISTRY


@tool
def load_skill(skill_name: str) -> str:
    """Load a specialized hiking or agent skill prompt and context.

    Call this when you need deep domain knowledge. Available skill
    names are returned in the error message if you request an unknown one.

    Returns the skill's system prompt and any reference context.
    """
    if not SKILL_REGISTRY:
        return "No skills are registered yet."

    skill = SKILL_REGISTRY.get(skill_name)
    if skill is None:
        known = sorted(SKILL_REGISTRY)
        return f"Unknown skill '{skill_name}'. Available skills: {', '.join(known)}"

    parts = [skill.prompt]
    for ref in skill.references:
        parts.append(f"\nReference: {ref}")
    return "\n".join(parts)
