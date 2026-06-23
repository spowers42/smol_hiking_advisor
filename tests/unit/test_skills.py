import pytest

from app.constants import build_system_prompt
from app.skills import SKILL_REGISTRY, Skill, build_skills_section
from app.skills.loader import load_skill


@pytest.fixture
def registered_skill():
    SKILL_REGISTRY["trail_analysis"] = Skill(
        prompt="You are a trail analysis expert.",
        description="Analyze trail difficulty, elevation, and distance.",
        references=["app/skills/prompts/trails.md"],
    )
    yield
    SKILL_REGISTRY.clear()


class TestSkill:
    def test_default_references_is_empty(self):
        skill = Skill(prompt="You are a test.")
        assert skill.references == []

    def test_can_set_references(self):
        skill = Skill(prompt="You are a test.", references=["ref1", "ref2"])
        assert skill.references == ["ref1", "ref2"]

    def test_default_description_is_empty(self):
        skill = Skill(prompt="You are a test.")
        assert skill.description == ""

    def test_can_set_description(self):
        skill = Skill(prompt="You are a test.", description="Expert trail analysis.")
        assert skill.description == "Expert trail analysis."


class TestBuildSkillsSection:
    def setup_method(self):
        SKILL_REGISTRY.clear()

    def test_returns_empty_when_registry_empty(self):
        assert build_skills_section() == ""

    def test_lists_skill_names(self, registered_skill):
        section = build_skills_section()
        assert "Available skills:" in section
        assert "trail_analysis" in section

    def test_includes_descriptions(self, registered_skill):
        section = build_skills_section()
        assert "Analyze trail difficulty" in section

    def test_sorts_alphabetically(self):
        SKILL_REGISTRY["z_skill"] = Skill(prompt="Z")
        SKILL_REGISTRY["a_skill"] = Skill(prompt="A")
        section = build_skills_section()
        assert section.index("a_skill") < section.index("z_skill")


class TestBuildSystemPrompt:
    def setup_method(self):
        SKILL_REGISTRY.clear()

    def test_includes_skills_section_when_skills_registered(self, registered_skill):
        prompt = build_system_prompt()
        assert "Available skills:" in prompt
        assert "trail_analysis" in prompt
        assert "Analyze trail difficulty" in prompt

    def test_omits_skills_section_when_no_skills(self):
        prompt = build_system_prompt()
        assert "Available skills:" not in prompt

    def test_still_contains_base_instructions(self):
        prompt = build_system_prompt()
        assert "get_hiker_profile" in prompt
        assert "load_skill" in prompt
        assert "get_current_conditions" in prompt


class TestLoadSkill:
    def setup_method(self):
        SKILL_REGISTRY.clear()

    def test_returns_empty_message_when_no_skills_registered(self):
        result = load_skill.invoke({"skill_name": "trail_analysis"})
        assert "No skills are registered" in result

    def test_returns_skill_prompt_from_registry(self, registered_skill):
        result = load_skill.invoke({"skill_name": "trail_analysis"})
        assert "You are a trail analysis expert." in result
        assert "app/skills/prompts/trails.md" in result

    def test_returns_error_for_unknown_skill(self):
        SKILL_REGISTRY["known_skill"] = Skill(prompt="You are known.")
        result = load_skill.invoke({"skill_name": "nonsense"})
        assert "Unknown skill" in result
        assert "known_skill" in result
