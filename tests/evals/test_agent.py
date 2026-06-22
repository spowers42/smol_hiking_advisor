import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, ToolCall

from app.agent import get_agent

from .metrics import ToolSelectionMetric, parse_tools_called

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

AGENT = None


def get_or_create_agent():
    global AGENT
    if AGENT is None:
        AGENT = get_agent()
    return AGENT


@pytest.mark.parametrize("golden", GOLDENS)
def test_tool_selection(golden):
    agent = get_or_create_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": golden["input"]}]})

    last_msg = result["messages"][-1]

    tc = LLMTestCase(
        input=golden["input"],
        actual_output=last_msg.content,
        tools_called=parse_tools_called(result),
        expected_tools=golden["expected_tools"],
    )
    assert_test(tc, metrics=[ToolSelectionMetric()])
