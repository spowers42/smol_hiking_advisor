from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.runtime import AgentRuntime, RuntimeConfig


class TestRuntimeConfig:
    def test_holds_fields(self):
        llm = MagicMock()
        config = RuntimeConfig(
            llm=llm,
            tools=[MagicMock()],
            prompt="You are helpful.",
            max_iterations=5,
        )
        assert config.llm is llm
        assert len(config.tools) == 1
        assert config.prompt == "You are helpful."
        assert config.max_iterations == 5

    def test_default_max_iterations(self):
        config = RuntimeConfig(llm=MagicMock())
        assert config.max_iterations == 10


class TestAgentRuntime:
    @patch("app.runtime.SimpleReActAgent")
    def test_invoke_delegates_to_agent(self, mock_simple_agent):
        mock_agent_instance = MagicMock()
        mock_agent_instance.invoke.return_value = {
            "messages": [
                HumanMessage(content="Good hike?"),
                AIMessage(content="Mount Major is great."),
            ],
        }
        mock_simple_agent.return_value = mock_agent_instance

        config = RuntimeConfig(llm=MagicMock(), tools=[], prompt="You are a hiking guide.")
        runtime = AgentRuntime(config)
        result = runtime.invoke({"messages": [("human", "Good hike?")]})

        assert result["messages"][-1].content == "Mount Major is great."
        mock_simple_agent.assert_called_once_with(
            llm=config.llm, tools=config.tools, prompt=config.prompt, max_iterations=10
        )

    @patch("app.runtime.SimpleReActAgent")
    def test_works_with_empty_tools(self, mock_simple_agent):
        mock_agent_instance = MagicMock()
        mock_agent_instance.invoke.return_value = {
            "messages": [
                HumanMessage(content="Best hike?"),
                AIMessage(content="Franconia Ridge is great."),
            ],
        }
        mock_simple_agent.return_value = mock_agent_instance

        config = RuntimeConfig(llm=MagicMock(), tools=[], prompt="You are helpful.")
        runtime = AgentRuntime(config)
        result = runtime.invoke({"messages": [("human", "Best hike?")]})

        assert "Franconia" in result["messages"][-1].content
