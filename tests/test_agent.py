from agent import parse_assistant_response


class TestParseAssistantResponse:
    def test_extracts_text_after_assistant_tag(self):
        raw = (
            "<|system|>You are a hiking advisor.<|end|>\n"
            "<|user|>What trails are good?<|end|>\n"
            "<|assistant|>Mount Major is a great option today.<|end|>"
        )
        result = parse_assistant_response(raw)
        assert result == "Mount Major is a great option today."
        assert "hiking advisor" not in result
        assert "What trails" not in result

    def test_handles_multiple_turns(self):
        raw = (
            "<|system|>You are a hiking advisor.<|end|>\n"
            "<|user|>Hi<|end|>\n"
            "<|assistant|>Hello!<|end|>\n"
            "<|user|>Best trail?<|end|>\n"
            "<|assistant|>The Franconia Ridge Loop is stunning.<|end|>"
        )
        result = parse_assistant_response(raw)
        assert result == "The Franconia Ridge Loop is stunning."
        assert "Hello" not in result
        assert "Best trail" not in result

    def test_returns_full_text_when_no_assistant_tag(self):
        raw = "Just a plain response without tags."
        assert parse_assistant_response(raw) == raw

    def test_handles_trailing_whitespace(self):
        raw = "<|assistant|>  A nice hike with extra spaces.  <|end|>"
        expected = "A nice hike with extra spaces."
        assert parse_assistant_response(raw) == expected

    def test_handles_empty_response(self):
        raw = "<|assistant|><|end|>"
        expected = ""
        assert parse_assistant_response(raw) == expected

    def test_handles_plain_text_with_tags_not_assistant(self):
        raw = "<|system|>Do not show this.<|end|>"
        assert parse_assistant_response(raw) == raw

    def test_no_closing_end_tag(self):
        raw = (
            "<|system|>You are a hiking advisor.<|end|>\n"
            "<|user|>hi, who are you?<|end|>\n"
            "<|assistant|>\n"
            "Hello! I'm Phi. I can help with hiking."  # no trailing <|end|>
        )
        result = parse_assistant_response(raw)
        assert result == "Hello! I'm Phi. I can help with hiking."
        assert "You are a hiking advisor" not in result
        assert "hi, who are you" not in result

    def test_handles_llama_format(self):
        raw = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "\n"
            "You are a hiking advisor.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "What trails are good?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            "\n"
            "Mount Major is a great option today."
        )
        result = parse_assistant_response(raw)
        assert result == "Mount Major is a great option today."
        assert "system" not in result
        assert "What trails" not in result
        assert "begin_of_text" not in result

    def test_handles_llama_format_with_closing_eot_id(self):
        raw = (
            "<|start_header_id|>system<|end_header_id|>\n"
            "\n"
            "System prompt here.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "Question?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            "\n"
            "The answer is here.<|eot_id|>"
        )
        result = parse_assistant_response(raw)
        assert result == "The answer is here."
        assert "System prompt here" not in result
        assert "Question" not in result
