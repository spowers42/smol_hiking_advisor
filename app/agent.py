import asyncio
import json
import re

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

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
            elif isinstance(msg, tuple):
                _role, content = msg
                messages.append(HumanMessage(content=content))
            else:
                messages.append(HumanMessage(content=msg["content"]))

        for _ in range(self.max_iterations):
            response = self.llm.invoke(messages, tools=list(self._tool_objects.values()))
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                tool = self._tool_objects.get(tc["name"])
                if tool:
                    try:
                        try:
                            asyncio.get_running_loop()
                            result = tool.invoke(tc["args"])
                        except RuntimeError:
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
