from langchain.tools import tool

_hiker_profile: dict[str, int | str] = {
    "fitness_level": 5,
    "experience_level": 5,
    "group_size": "Solo",
}


def set_hiker_profile(fitness: int, experience: int, group_size: str) -> None:
    _hiker_profile["fitness_level"] = fitness
    _hiker_profile["experience_level"] = experience
    _hiker_profile["group_size"] = group_size


@tool
def get_hiker_profile() -> dict[str, int | str]:
    """Get the current hiker profile (fitness level, experience level, and group size).

    Returns:
        A dictionary with keys: fitness_level (1-10), experience_level (1-10),
        and group_size ("Solo", "2–3", or "3+").
    """
    return dict(_hiker_profile)
