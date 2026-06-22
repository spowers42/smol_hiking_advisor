from app.response_parser import parse_model_output, parse_tool_call


class TestParseToolCall:
    def test_extracts_name_and_json_args(self):
        result = parse_tool_call('{"function": "get_weather", "arguments": {"loc": "Summit"}}')
        assert result == ("get_weather", {"loc": "Summit"})

    def test_extracts_name_no_args(self):
        result = parse_tool_call('{"function": "get_weather"}')
        assert result == ("get_weather", {})

    def test_extracts_with_string_args(self):
        text = '{"function": "get_weather", "arguments": "{\\"loc\\": \\"Summit\\"}"}'
        result = parse_tool_call(text)
        assert result is not None
        assert result[0] == "get_weather"

    def test_returns_none_on_no_match(self):
        assert parse_tool_call("Hello, how are you?") is None

    def test_returns_none_on_empty_string(self):
        assert parse_tool_call("") is None

    def test_handles_malformed_json_args(self):
        result = parse_tool_call('{"function": "foo", "arguments": "{broken}')
        assert result is None or result == ("foo", {})

    def test_handles_extra_text_around_call(self):
        text = 'Some preamble {"function": "get_weather", "arguments": {}} trailing text'
        result = parse_tool_call(text)
        assert result == ("get_weather", {})


class TestParseModelOutput:
    def test_returns_parsed_output_with_valid_tool(self):
        result = parse_model_output(
            '{"function": "get_weather", "arguments": {"loc": "Summit"}}',
            {"get_weather"},
        )
        assert result.text == '{"function": "get_weather", "arguments": {"loc": "Summit"}}'
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["args"] == {"loc": "Summit"}

    def test_ignores_hallucinated_tool(self):
        result = parse_model_output(
            '{"function": "suggest_hike", "arguments": {}}',
            {"get_weather"},
        )
        assert len(result.tool_calls) == 0
        assert result.text == '{"function": "suggest_hike", "arguments": {}}'

    def test_returns_empty_tool_calls_for_plain_text(self):
        result = parse_model_output("Mount Major is a great hike!", {"get_weather"})
        assert len(result.tool_calls) == 0
        assert result.text == "Mount Major is a great hike!"

    def test_handles_empty_valid_tools_set(self):
        result = parse_model_output(
            '{"function": "get_weather", "arguments": {}}',
            set(),
        )
        assert len(result.tool_calls) == 0

    def test_unique_call_ids(self):
        r1 = parse_model_output(
            '{"function": "get_weather", "arguments": {}}',
            {"get_weather"},
        )
        r2 = parse_model_output(
            '{"function": "get_weather", "arguments": {}}',
            {"get_weather"},
        )
        assert r1.tool_calls[0]["id"] != r2.tool_calls[0]["id"]
