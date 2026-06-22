from unittest.mock import MagicMock

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
    def test_invoke_delegates_to_agent(self):
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="Mount Major is great.")
        config = RuntimeConfig(llm=llm, tools=[], prompt="You are a hiking guide.")
        runtime = AgentRuntime(config)
        result = runtime.invoke({"messages": [HumanMessage(content="Good hike?")]})
        assert "Mount Major" in result["messages"][-1].content

    def test_works_with_empty_tools(self):
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="Try the Franconia Ridge Loop.")
        config = RuntimeConfig(llm=llm, prompt="You are helpful.")
        runtime = AgentRuntime(config)
        result = runtime.invoke({"messages": [HumanMessage(content="Best hike?")]})
        assert "Franconia Ridge" in result["messages"][-1].content
