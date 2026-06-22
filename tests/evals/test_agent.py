import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, ToolCall

from app import constants
from app.config import get_llm
from app.runtime import AgentRuntime, RuntimeConfig
from app.tools.user_preferences import get_user_preferences
from app.tools.weather import connect as connect_weather_mcp
from app.tools.weather import get_weather_tools

from .metrics import ToolSelectionMetric, parse_tools_called

WEATHER_TOOL_NAMES = {"get_current_conditions", "get_summit_forecast"}

GOLDENS = [
    {
        "input": "What's a good hike for me today?",
        "expected_tools": [ToolCall(name="get_user_preferences")],
    },
    {
        "input": "What's the current weather on top of Mt Washington?",
        "expected_tools": [
            ToolCall(name="get_current_conditions"),
            ToolCall(name="get_summit_forecast"),
        ],
    },
    {
        "input": "I'm experienced and fit, looking for a challenging hike in the White Mountains",
        "expected_tools": [ToolCall(name="get_user_preferences")],
    },
]


def build_agent() -> AgentRuntime:
    connect_weather_mcp()
    llm = get_llm()
    tools = [get_user_preferences, *get_weather_tools()]
    config = RuntimeConfig(llm=llm, tools=tools, prompt=constants.SYSTEM_PROMPT)
    return AgentRuntime(config)


@pytest.mark.parametrize("golden", GOLDENS)
def test_tool_selection(golden):
    agent = build_agent()

    weather_available = any(t.name in WEATHER_TOOL_NAMES for t in get_weather_tools())
    needs_weather = any(tc.name in WEATHER_TOOL_NAMES for tc in golden["expected_tools"])
    if needs_weather and not weather_available:
        pytest.skip("MCP weather server not available")

    result = agent.invoke({"messages": [{"role": "user", "content": golden["input"]}]})

    tc = LLMTestCase(
        input=golden["input"],
        actual_output=result["messages"][-1].content,
        tools_called=parse_tools_called(result),
        expected_tools=golden["expected_tools"],
    )
    assert_test(tc, metrics=[ToolSelectionMetric()])
