import pytest

from app.tools.user_preferences import get_user_preferences, set_preferences


@pytest.fixture(autouse=True)
def reset_prefs():
    set_preferences(fitness=5, experience=5, group_size="Solo")
    yield


class TestGetUserPreferencesTool:
    def test_returns_dict_with_expected_keys(self):
        result = get_user_preferences.invoke({})
        assert isinstance(result, dict)
        assert "fitness_level" in result
        assert "experience_level" in result
        assert "group_size" in result

    def test_has_default_values(self):
        result = get_user_preferences.invoke({})
        assert result["fitness_level"] == 5
        assert result["experience_level"] == 5
        assert result["group_size"] == "Solo"

    def test_set_preferences_updates_values(self):
        set_preferences(fitness=8, experience=3, group_size="2–3")
        result = get_user_preferences.invoke({})
        assert result["fitness_level"] == 8
        assert result["experience_level"] == 3
        assert result["group_size"] == "2–3"

    def test_set_preferences_at_boundaries(self):
        set_preferences(fitness=1, experience=10, group_size="3+")
        result = get_user_preferences.invoke({})
        assert result["fitness_level"] == 1
        assert result["experience_level"] == 10
        assert result["group_size"] == "3+"

    def test_get_user_preferences_returns_copy_not_reference(self):
        result = get_user_preferences.invoke({})
        result["fitness_level"] = 99
        unchanged = get_user_preferences.invoke({})
        assert unchanged["fitness_level"] == 5

    def test_tool_name_and_description(self):
        assert get_user_preferences.name == "get_user_preferences"
        assert "fitness level" in get_user_preferences.description.lower()
        assert "experience level" in get_user_preferences.description.lower()
        assert "group size" in get_user_preferences.description.lower()
