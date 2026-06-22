from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.agent import SimpleReActAgent


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
