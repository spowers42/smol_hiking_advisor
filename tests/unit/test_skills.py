import tempfile
from pathlib import Path

import pytest

from app.constants import build_system_prompt
from app.skills import Skill, build_skills_section, clear_registry, seed_registry
from app.skills.loader import load_skill


@pytest.fixture(autouse=True)
def reset_registry():
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def registered_skill():
    seed_registry(
        {
            "trail_analysis": Skill(
                name="trail_analysis",
                description="Analyze trail difficulty, elevation, and distance.",
            ),
        }
    )
    yield


@pytest.fixture
def skill_on_disk():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "trail-analysis"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            "---\n"
            "name: trail-analysis\n"
            "description: Analyze trail difficulty, elevation, and distance.\n"
            "---\n"
            "\n"
            "You are a trail analysis expert."
        )
        seed_registry(
            {
                "trail-analysis": Skill(
                    name="trail-analysis",
                    description="Analyze trail difficulty, elevation, and distance.",
                    path=str(skill_dir),
                ),
            }
        )
        yield


class TestSkill:
    def test_default_references_is_empty(self):
        skill = Skill(name="test")
        assert skill.references == []

    def test_can_set_references(self):
        skill = Skill(name="test", references=["ref1", "ref2"])
        assert skill.references == ["ref1", "ref2"]

    def test_default_description_is_empty(self):
        skill = Skill(name="test")
        assert skill.description == ""

    def test_can_set_description(self):
        skill = Skill(name="test", description="Expert trail analysis.")
        assert skill.description == "Expert trail analysis."


class TestBuildSkillsSection:
    def test_returns_empty_when_registry_empty(self):
        seed_registry({})
        assert build_skills_section() == ""

    def test_lists_skill_names(self, registered_skill):
        section = build_skills_section()
        assert "Available skills:" in section
        assert "trail_analysis" in section

    def test_includes_descriptions(self, registered_skill):
        section = build_skills_section()
        assert "Analyze trail difficulty" in section

    def test_sorts_alphabetically(self):
        seed_registry(
            {
                "z_skill": Skill(name="z_skill"),
                "a_skill": Skill(name="a_skill"),
            }
        )
        section = build_skills_section()
        assert section.index("a_skill") < section.index("z_skill")


class TestBuildSystemPrompt:
    def test_includes_skills_section_when_skills_registered(self, registered_skill):
        prompt = build_system_prompt()
        assert "Available skills:" in prompt
        assert "trail_analysis" in prompt
        assert "Analyze trail difficulty" in prompt

    def test_omits_skills_section_when_no_skills(self):
        seed_registry({})
        prompt = build_system_prompt()
        assert "Available skills:" not in prompt

    def test_still_contains_base_instructions(self):
        prompt = build_system_prompt()
        assert "get_hiker_profile" in prompt
        assert "load_skill" in prompt
        assert "get_current_conditions" in prompt


class TestLoadSkill:
    def test_returns_empty_message_when_no_skills_registered(self):
        seed_registry({})
        result = load_skill.invoke({"skill_name": "trail_analysis"})
        assert "No skills are registered" in result

    def test_returns_skill_body_from_disk(self, skill_on_disk):
        result = load_skill.invoke({"skill_name": "trail-analysis"})
        assert "You are a trail analysis expert." in result

    def test_returns_error_for_unknown_skill(self):
        seed_registry(
            {
                "known_skill": Skill(name="known_skill"),
            }
        )
        result = load_skill.invoke({"skill_name": "nonsense"})
        assert "Unknown skill" in result
        assert "known_skill" in result
