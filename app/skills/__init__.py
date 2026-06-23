from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_SKILLS_DIR = Path(__file__).parent


@dataclass
class Skill:
    name: str
    description: str = ""
    path: str = ""
    references: list[str] = field(default_factory=list)


def _parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}
    result: dict[str, str] = {}
    for line in lines[1:end]:
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result


def _discover_skills(base_dir: str | Path | None = None) -> dict[str, Skill]:
    root = Path(base_dir) if base_dir else _DEFAULT_SKILLS_DIR
    skills: dict[str, Skill] = {}
    for entry in sorted(os.listdir(root)):
        skill_dir = root / entry
        skill_file = skill_dir / "SKILL.md"
        if not skill_dir.is_dir() or not skill_file.exists():
            continue
        text = skill_file.read_text(encoding="utf-8")
        fm = _parse_frontmatter(text)
        name = fm.get("name", entry)
        description = fm.get("description", "")
        skills[name] = Skill(
            name=name,
            description=description,
            path=str(skill_dir),
        )
    return skills


_SKILL_REGISTRY: dict[str, Skill] | None = None


def get_registry() -> dict[str, Skill]:
    global _SKILL_REGISTRY
    if _SKILL_REGISTRY is None:
        _SKILL_REGISTRY = _discover_skills()
    return _SKILL_REGISTRY


def clear_registry() -> None:
    global _SKILL_REGISTRY
    _SKILL_REGISTRY = None


def seed_registry(skills: dict[str, Skill]) -> None:
    global _SKILL_REGISTRY
    _SKILL_REGISTRY = dict(skills)


def build_skills_section() -> str:
    registry = get_registry()
    if not registry:
        return ""
    lines = ["Available skills:"]
    for name in sorted(registry):
        skill = registry[name]
        if skill.description:
            lines.append(f"  {name} — {skill.description}")
        else:
            lines.append(f"  {name}")
    return "\n".join(lines)
