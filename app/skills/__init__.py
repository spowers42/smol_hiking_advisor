from dataclasses import dataclass, field


@dataclass
class SkillDef:
    prompt: str
    description: str = ""
    references: list[str] = field(default_factory=list)


SKILL_REGISTRY: dict[str, SkillDef] = {}


def build_skills_section() -> str:
    if not SKILL_REGISTRY:
        return ""
    lines = ["Available skills:"]
    for name in sorted(SKILL_REGISTRY):
        skill = SKILL_REGISTRY[name]
        if skill.description:
            lines.append(f"  {name} — {skill.description}")
        else:
            lines.append(f"  {name}")
    return "\n".join(lines)
