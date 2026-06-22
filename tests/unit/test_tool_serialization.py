from langchain_core.tools import tool
from langchain_core.tools.render import render_text_description_and_args

from app.models.tool_serialization import parse_tool_call


@tool
def _dummy_tool(location: str) -> str:
    """Get weather for a location."""
    return "sunny"


@tool
def _another_tool(count: int) -> str:
    """Get a count."""
    return "5"


class TestSerializeTools:
    def test_renders_tool_name_and_description(self):
        result = render_text_description_and_args([_dummy_tool])
        assert "get_weather" in result or "_dummy_tool" in result
        assert "location" in result

    def test_includes_multiple_tools(self):
        result = render_text_description_and_args([_dummy_tool, _another_tool])
        assert "location" in result
        assert "count" in result

    def test_handles_empty_tool_list(self):
        result = render_text_description_and_args([])
        assert result == ""


class TestParseToolCall:
    def test_parses_simple_tool_call(self):
        text = '{"function": "get_weather", "arguments": {"location": "summit"}}'
        result = parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "get_weather"
        assert args == {"location": "summit"}

    def test_parses_tool_call_without_arguments(self):
        text = '{"function": "get_weather"}'
        result = parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "get_weather"
        assert args == {}

    def test_parses_tool_call_with_extra_whitespace(self):
        text = '{  "function"  :  "get_weather"  ,  "arguments"  :  {"loc": "summit"}  }'
        result = parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "get_weather"
        assert args == {"loc": "summit"}

    def test_returns_none_for_plain_text(self):
        text = "The weather on Mt Washington is 23°F with winds at 40mph."
        assert parse_tool_call(text) is None

    def test_returns_none_for_empty_string(self):
        assert parse_tool_call("") is None

    def test_returns_none_for_malformed_json(self):
        text = '{"function": "get_weather", "arguments": {broken}}'
        result = parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "get_weather"
        assert args == {}

    def test_handles_string_arguments(self):
        text = '{"function": "get_weather", "arguments": "summit"}'
        result = parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "get_weather"
        assert args == {}

    def test_parses_tool_call_in_middle_of_text(self):
        text = (
            "Thought: I should check the weather.\n"
            '{"function": "get_weather", "arguments": {"loc": "summit"}}\n'
            "Observation: 23°F"
        )
        result = parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "get_weather"
        assert args == {"loc": "summit"}

    def test_parses_tool_call_with_nested_quotes(self):
        text = '{"function": "get_weather", "arguments": {"loc": "summit"}}'
        result = parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "get_weather"
        assert args == {"loc": "summit"}
