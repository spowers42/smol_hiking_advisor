from typing import List

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall
from langchain_core.messages import AIMessage


def parse_tools_called(result: dict) -> List[ToolCall]:
    tools = []
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tools.append(ToolCall(name=tc["name"]))
    return tools


class ToolSelectionMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self._required_params = [
            SingleTurnParams.INPUT,
            SingleTurnParams.TOOLS_CALLED,
            SingleTurnParams.EXPECTED_TOOLS,
        ]

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        called_names = {tc.name for tc in (test_case.tools_called or [])}
        expected_names = {tc.name for tc in (test_case.expected_tools or [])}

        if not expected_names:
            self.score = 1.0
            self.reason = "No expected tools — skipping evaluation."
            self.success = True
            return self.score

        match_count = len(called_names & expected_names)
        self.score = match_count / len(expected_names)
        self.success = self.score >= self.threshold

        details = []
        for t in expected_names:
            if t in called_names:
                details.append(f"✓ {t}")
            else:
                details.append(f"✗ {t} (missing)")

        self.reason = f"Tool recall: {match_count}/{len(expected_names)}. {'; '.join(details)}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "ToolSelection"
