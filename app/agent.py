import asyncio
import json
import re

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app import constants
from app.config import get_llm
from app.models.tool_calling_model import ToolCallingChatModel
from app.tools.user_preferences import get_user_preferences
from app.tools.weather import connect as connect_weather_mcp
from app.tools.weather import get_weather_tools


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


def _tool_dict(t):
    try:
        params = t.tool_call_schema.model_json_schema()
    except Exception:
        params = {}
    return {
        "name": t.name,
        "description": t.description,
        "parameters": params,
    }


class SimpleReActAgent:
    def __init__(self, llm, tools, prompt, max_iterations=10):
        self.llm = llm
        self._tool_objects = {t.name: t for t in tools}
        self.prompt = prompt
        self.max_iterations = max_iterations

    def invoke(self, input_dict):
        messages = [SystemMessage(content=self.prompt)]
        for msg in input_dict["messages"]:
            if isinstance(msg, BaseMessage):
                messages.append(msg)
            else:
                messages.append(HumanMessage(content=msg["content"]))

        tool_dicts = [_tool_dict(t) for t in self._tool_objects.values()]

        for _ in range(self.max_iterations):
            response = self.llm.invoke(messages, tools=tool_dicts)
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                tool = self._tool_objects.get(tc["name"])
                if tool:
                    try:
                        result = asyncio.run(tool.ainvoke(tc["args"]))
                    except Exception as e:
                        result = {"error": str(e)}
                else:
                    result = {"error": f"Unknown tool: {tc['name']}"}
                messages.append(
                    ToolMessage(
                        content=json.dumps(result),
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                )

        return {"messages": messages}


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        connect_weather_mcp()
        llm = ToolCallingChatModel(chat=get_llm())
        tools = [get_user_preferences, *get_weather_tools()]
        _agent = SimpleReActAgent(llm=llm, tools=tools, prompt=constants.SYSTEM_PROMPT)
    return _agent
