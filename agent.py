import re

from langgraph.prebuilt import create_react_agent

import constants
from config import get_llm
from user_preferences import get_user_preferences

ASSISTANT_PATTERN = re.compile(
    r"(?:<\|assistant\|>|<\|start_header_id\|>assistant<\|end_header_id\|>)"
    r"(.*?)(?:<\|end\|>|<\|eot_id\|>|$)",
    re.DOTALL,
)

FINAL_ANSWER_PATTERN = re.compile(r".*?Final Answer:\s*(.*)", re.DOTALL)


def parse_assistant_response(raw: str) -> str:
    matches = ASSISTANT_PATTERN.findall(raw)
    text = matches[-1].strip() if matches else raw
    final_match = FINAL_ANSWER_PATTERN.fullmatch(text)
    if final_match:
        return final_match.group(1).strip()
    return text


_agent_executor = None


def get_agent():
    global _agent_executor
    if _agent_executor is None:
        llm = get_llm()
        _agent_executor = create_react_agent(
            llm,
            tools=[get_user_preferences],
            prompt=constants.SYSTEM_PROMPT,
        )
    return _agent_executor
