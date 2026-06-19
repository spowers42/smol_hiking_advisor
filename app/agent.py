import re

from langgraph.prebuilt import create_react_agent

from app import constants
from app.config import get_llm

ASSISTANT_PATTERN = re.compile(
    r"(?:<\|assistant\|>|<\|start_header_id\|>assistant<\|end_header_id\|>)"
    r"(.*?)(?:<\|end\|>|<\|eot_id\|>|$)",
    re.DOTALL,
)


def parse_assistant_response(raw: str) -> str:
    matches = ASSISTANT_PATTERN.findall(raw)
    if not matches:
        return raw
    text = matches[-1].strip()
    return text


_agent_executor = None


def get_agent():
    global _agent_executor
    if _agent_executor is None:
        llm = get_llm()
        _agent_executor = create_react_agent(
            llm,
            tools=[],
            prompt=constants.SYSTEM_PROMPT,
        )
    return _agent_executor
