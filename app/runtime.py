import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages.utils import convert_to_messages

from app.agent import SimpleReActAgent

logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    llm: Any
    tools: list = field(default_factory=list)
    prompt: str = ""
    max_iterations: int = 15


class AgentRuntime:
    def __init__(self, config: RuntimeConfig):
        self._agent = SimpleReActAgent(
            llm=config.llm,
            tools=config.tools,
            prompt=config.prompt,
            max_iterations=config.max_iterations,
        )
        logger.info(
            "AgentRuntime created with %d tool(s), max_iterations=%d",
            len(config.tools),
            config.max_iterations,
        )

    def invoke(self, input_dict: dict) -> dict:
        raw_messages = input_dict.get("messages", [])
        messages = convert_to_messages(raw_messages)
        return self._agent.invoke({"messages": messages})
