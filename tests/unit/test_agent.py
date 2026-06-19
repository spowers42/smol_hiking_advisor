from app.agent import parse_assistant_response


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

    def test_strips_final_answer_prefix(self):
        raw = "<|assistant|>Final Answer: Try Mount Major.<|end|>"
        result = parse_assistant_response(raw)
        assert result == "Try Mount Major."

    def test_strips_thought_and_final_answer(self):
        raw = (
            "<|assistant|>"
            "Thought: I have the hiker's profile.\n"
            "Final Answer: The Franconia Ridge Loop is a great fit.\n"
            "<|end|>"
        )
        result = parse_assistant_response(raw)
        assert result == "The Franconia Ridge Loop is a great fit."

    def test_strips_multiline_final_answer(self):
        raw = (
            "<|assistant|>Final Answer: Here are some hikes:\n"
            "- Mount Major\n- Franconia Ridge<|end|>"
        )
        result = parse_assistant_response(raw)
        expected = "Here are some hikes:\n- Mount Major\n- Franconia Ridge"
        assert result == expected

    def test_passes_through_plain_text_without_final_answer(self):
        raw = "<|assistant|>Just a regular response.<|end|>"
        result = parse_assistant_response(raw)
        assert result == "Just a regular response."

    def test_strips_final_answer_from_tag_stripped_text(self):
        raw = (
            "Thought: I have the hiker's profile. "
            "Given your high fitness, I should suggest a challenging trail.\n"
            "Final Answer: The Presidential Traverse is a great option."
        )
        result = parse_assistant_response(raw)
        assert result == "The Presidential Traverse is a great option."

    def test_strips_react_format_with_question_and_action(self):
        raw = (
            "Question: You're looking for a challenging hike?\n"
            "Thought: I should check preferences first.\n"
            "Action: get_user_preferences\n"
            "Action Input: {}\n"
            'Observation: {"fitness_level": 8, "experience_level": 8, "group_size": "Solo"}\n'
            "Thought: Now I have the profile.\n"
            "Final Answer: Mount Major is a great option."
        )
        result = parse_assistant_response(raw)
        assert result == "Mount Major is a great option."
