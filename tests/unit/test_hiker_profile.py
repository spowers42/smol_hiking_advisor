import pytest

from app.tools.hiker_profile import get_hiker_profile, set_hiker_profile


@pytest.fixture(autouse=True)
def reset_profile():
    set_hiker_profile(fitness=5, experience=5, group_size="Solo")
    yield


class TestGetHikerProfileTool:
    def test_returns_dict_with_expected_keys(self):
        result = get_hiker_profile.invoke({})
        assert isinstance(result, dict)
        assert "fitness_level" in result
        assert "experience_level" in result
        assert "group_size" in result

    def test_has_default_values(self):
        result = get_hiker_profile.invoke({})
        assert result["fitness_level"] == 5
        assert result["experience_level"] == 5
        assert result["group_size"] == "Solo"

    def test_set_hiker_profile_updates_values(self):
        set_hiker_profile(fitness=8, experience=3, group_size="2–3")
        result = get_hiker_profile.invoke({})
        assert result["fitness_level"] == 8
        assert result["experience_level"] == 3
        assert result["group_size"] == "2–3"

    def test_set_hiker_profile_at_boundaries(self):
        set_hiker_profile(fitness=1, experience=10, group_size="3+")
        result = get_hiker_profile.invoke({})
        assert result["fitness_level"] == 1
        assert result["experience_level"] == 10
        assert result["group_size"] == "3+"

    def test_get_hiker_profile_returns_copy_not_reference(self):
        result = get_hiker_profile.invoke({})
        result["fitness_level"] = 99
        unchanged = get_hiker_profile.invoke({})
        assert unchanged["fitness_level"] == 5

    def test_tool_name_and_description(self):
        assert get_hiker_profile.name == "get_hiker_profile"
        assert "fitness level" in get_hiker_profile.description.lower()
        assert "experience level" in get_hiker_profile.description.lower()
        assert "group size" in get_hiker_profile.description.lower()
