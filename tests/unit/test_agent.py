from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from app.agent import (
    SimpleReActAgent,
    _patch_openai_content_stripping,
    _strip_tool_call_content,
)


class TestStripToolCallContent:
    _LLAMA_TOOL = {
        "name": "load_skill",
        "args": {"skill_name": "trail-safety"},
        "id": "1",
        "type": "tool_call",
    }
    _LLAMA_TOOL_2 = {
        "name": "load_skill",
        "args": {"skill_name": "trail-safety"},
        "id": "2",
        "type": "tool_call",
    }
    _GET_CONDITIONS = {"name": "get_conditions", "args": {}, "id": "1", "type": "tool_call"}
    _LOAD_SKILL_EMPTY = {"name": "load_skill", "args": {}, "id": "1", "type": "tool_call"}

    def test_clears_content_when_tool_calls_and_json_content(self):
        msg = AIMessage(
            content='; {"name": "load_skill", "parameters": {"skill_name": "trail-safety"}}',
            tool_calls=[self._LLAMA_TOOL],
        )
        result = ChatResult(generations=[ChatGeneration(message=msg)])
        cleaned = _strip_tool_call_content(result)
        assert cleaned.generations[0].message.content == ""

    def test_preserves_content_when_no_tool_calls(self):
        msg = AIMessage(content="Nice day for a hike!")
        result = ChatResult(generations=[ChatGeneration(message=msg)])
        cleaned = _strip_tool_call_content(result)
        assert cleaned.generations[0].message.content == "Nice day for a hike!"

    def test_preserves_content_when_no_json_pattern(self):
        msg = AIMessage(
            content="I'll look up the trail conditions.",
            tool_calls=[self._GET_CONDITIONS],
        )
        result = ChatResult(generations=[ChatGeneration(message=msg)])
        cleaned = _strip_tool_call_content(result)
        assert cleaned.generations[0].message.content == "I'll look up the trail conditions."

    def test_clears_duplicate_tool_calls_content(self):
        msg = AIMessage(
            content=(
                '; {"name": "load_skill", "parameters": {"skill_name": "trail-safety"}};'
                ' {"name": "load_skill", "parameters": {"skill_name": "trail-safety"}}'
            ),
            tool_calls=[self._LLAMA_TOOL, self._LLAMA_TOOL_2],
        )
        result = ChatResult(generations=[ChatGeneration(message=msg)])
        cleaned = _strip_tool_call_content(result)
        assert cleaned.generations[0].message.content == ""

    def test_clears_function_format_content(self):
        msg = AIMessage(
            content='{"function": "load_skill", "arguments": {"skill_name": "trail-safety"}}',
            tool_calls=[self._LLAMA_TOOL],
        )
        result = ChatResult(generations=[ChatGeneration(message=msg)])
        cleaned = _strip_tool_call_content(result)
        assert cleaned.generations[0].message.content == ""

    def test_handles_non_ai_message(self):
        msg = ToolMessage(content="result", tool_call_id="1")
        result = ChatResult(generations=[ChatGeneration(message=msg)])
        cleaned = _strip_tool_call_content(result)
        assert cleaned.generations[0].message.content == "result"

    def test_preserves_empty_content_with_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[self._LOAD_SKILL_EMPTY],
        )
        result = ChatResult(generations=[ChatGeneration(message=msg)])
        cleaned = _strip_tool_call_content(result)
        assert cleaned.generations[0].message.content == ""

    def test_preserves_content_with_tool_calls_no_json_pattern(self):
        msg = AIMessage(
            content="I found the trail info.",
            tool_calls=[self._LLAMA_TOOL],
        )
        result = ChatResult(generations=[ChatGeneration(message=msg)])
        cleaned = _strip_tool_call_content(result)
        assert cleaned.generations[0].message.content == "I found the trail info."


class TestPatchOpenAIContentStripping:
    _EMPTY_TOOL = {"name": "load_skill", "args": {}, "id": "1", "type": "tool_call"}

    def test_patches_generate_method(self):
        model = MagicMock(spec=ChatOpenAI)
        original = model._generate
        _patch_openai_content_stripping(model)
        assert model._generate is not original
        msg = AIMessage(
            content='{"name": "load_skill", "parameters": {"skill_name": "trail-safety"}}',
            tool_calls=[self._EMPTY_TOOL],
        )
        original.return_value = ChatResult(generations=[ChatGeneration(message=msg)])
        result = model._generate([])
        assert isinstance(result, ChatResult)
        assert result.generations[0].message.content == ""


class TestSimpleReActAgent:
    @patch("app.agent.create_agent")
    def test_invoke_delegates_to_create_agent(self, mock_create_agent):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [
                HumanMessage(content="Good hike?"),
                AIMessage(content="Mount Major is great."),
            ],
        }
        mock_create_agent.return_value = mock_agent

        agent = SimpleReActAgent(llm=MagicMock(), tools=[], prompt="You are a hiking guide.")
        result = agent.invoke({"messages": [("human", "Good hike?")]})

        assert result["messages"][-1].content == "Mount Major is great."
        mock_create_agent.assert_called_once()
        mock_agent.invoke.assert_called_once()

    @patch("app.agent.create_agent")
    def test_creates_agent_with_llm_and_tools(self, mock_create_agent):
        llm = MagicMock()
        tool = MagicMock()
        tool.name = "get_weather"

        SimpleReActAgent(llm=llm, tools=[tool], prompt="You are helpful.")

        mock_create_agent.assert_called_once()
        _, kwargs = mock_create_agent.call_args
        assert kwargs["model"] is llm
        assert kwargs["tools"] == [tool]
        assert kwargs["system_prompt"] == "You are helpful."

    @patch("app.agent.create_agent")
    def test_passes_recursion_limit_from_max_iterations(self, mock_create_agent):
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        agent = SimpleReActAgent(
            llm=MagicMock(), tools=[], prompt="You are helpful.", max_iterations=5
        )
        agent.invoke({"messages": [("human", "Hi")]})

        _, kwargs = mock_agent.invoke.call_args
        assert kwargs["config"]["recursion_limit"] == 15  # 5 * 2 + 5

    @patch("app.agent.create_agent")
    def test_returns_tool_calls_in_messages(self, mock_create_agent):
        tool = MagicMock()
        tool.name = "get_weather"

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [
                HumanMessage(content="Weather?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "get_weather", "args": {}, "id": "1", "type": "tool_call"}
                    ],
                ),
            ],
        }
        mock_create_agent.return_value = mock_agent

        agent = SimpleReActAgent(llm=MagicMock(), tools=[tool], prompt="You are helpful.")
        result = agent.invoke({"messages": [("human", "Weather?")]})

        ai_msg = result["messages"][-1]
        assert len(ai_msg.tool_calls) == 1
        assert ai_msg.tool_calls[0]["name"] == "get_weather"

    @patch("app.agent.create_agent")
    def test_works_with_empty_tools(self, mock_create_agent):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [
                HumanMessage(content="Best hike?"),
                AIMessage(content="Franconia Ridge is great."),
            ],
        }
        mock_create_agent.return_value = mock_agent

        agent = SimpleReActAgent(llm=MagicMock(), tools=[], prompt="You are a hiking guide.")
        result = agent.invoke({"messages": [("human", "Best hike?")]})
        assert "Franconia" in result["messages"][-1].content
