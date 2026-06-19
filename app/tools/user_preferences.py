from langchain.tools import tool

_user_preferences: dict[str, int | str] = {
    "fitness_level": 5,
    "experience_level": 5,
    "group_size": "Solo",
}


def set_preferences(fitness: int, experience: int, group_size: str) -> None:
    _user_preferences["fitness_level"] = fitness
    _user_preferences["experience_level"] = experience
    _user_preferences["group_size"] = group_size


@tool
def get_user_preferences() -> dict[str, int | str]:
    """Get the current user preferences for fitness level, experience level, and group size.

    Returns:
        A dictionary with keys: fitness_level (1-10), experience_level (1-10),
        and group_size ("Solo", "2–3", or "3+").
    """
    return dict(_user_preferences)
