import asyncio
import json
import re
from typing import Any, List, Optional
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langchain_core.tools.render import render_text_description_and_args
from langchain_huggingface import ChatHuggingFace

_FN_CALL_RE = re.compile(
    r'\{\s*"function"\s*:\s*"([^"]+)"(?:\s*,\s*(?:"arguments"\s*:\s*)?(\{.*?\}|".*?"))?\s*\}',
    re.DOTALL,
)


def parse_tool_call(text: str) -> Optional[tuple[str, dict]]:
    match = _FN_CALL_RE.search(text)
    if not match:
        return None
    name = match.group(1)
    args_raw = match.group(2)
    if args_raw:
        if args_raw.startswith('"') and args_raw.endswith('"'):
            inner = args_raw[1:-1]
            try:
                args = json.loads(inner) if inner else {}
            except json.JSONDecodeError:
                args = {}
        else:
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {}
    else:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return (name, args)


def _inject_tool_descriptions(messages: List[BaseMessage], tools: List) -> List[BaseMessage]:
    messages = list(messages)
    rendered = render_text_description_and_args(tools)
    instructions = (
        "\n\nYou have access to the following functions. Use them as needed.\n\n"
        "When you need to call a function, respond with ONLY a JSON object "
        'on its own line in this format:\n{"function": "name", "arguments": {}}\n'
        "Call one function at a time.\n\n"
        "After you receive the tool result, provide your answer. "
        "Do not call the same function twice."
    )
    block = rendered + "\n\n" + instructions
    for i, m in enumerate(messages):
        if isinstance(m, SystemMessage):
            content = m.content
            if "You have access to the following functions" not in content:
                messages[i] = SystemMessage(content=content + "\n\n" + block)
            return messages
    messages.insert(0, SystemMessage(content=block))
    return messages


def _parse_text_tool_calls(result: ChatResult, valid_tool_names: set) -> ChatResult:
    generations = []
    for gen in result.generations:
        text = gen.message.content
        parsed = parse_tool_call(text)
        if parsed and parsed[0] in valid_tool_names:
            name, args = parsed
            msg = AIMessage(
                content=text,
                tool_calls=[
                    ToolCall(name=name, args=args, id=f"call_{uuid4().hex[:12]}")
                ],
            )
            generations.append(ChatGeneration(message=msg))
        else:
            generations.append(gen)
    return ChatResult(generations=generations, llm_output=result.llm_output)


def _needs_local_tool_handling(llm: Any) -> bool:
    if not isinstance(llm, ChatHuggingFace):
        return False
    from langchain_huggingface.chat_models.huggingface import _is_huggingface_endpoint, _is_huggingface_textgen_inference
    return not (_is_huggingface_endpoint(llm.llm) or _is_huggingface_textgen_inference(llm.llm))


def wrap_model_for_tools(model: Any, tools: List) -> Any:
    if not tools or not _needs_local_tool_handling(model):
        return model

    valid_tool_names = {t.name for t in tools}
    original_generate = model._generate
    original_to_chatml = model._to_chatml_format
    original_to_chat_prompt = model._to_chat_prompt

    def _patched_to_chatml(message):
        if isinstance(message, ToolMessage):
            return {"role": "user", "content": f"Tool result ({message.name}): {message.content}"}
        return original_to_chatml(message)

    model._to_chatml_format = _patched_to_chatml

    def _patched_to_chat_prompt(messages):
        if not messages:
            raise ValueError("At least one message must be provided!")
        message_dicts = [_patched_to_chatml(m) for m in messages]
        return model.tokenizer.apply_chat_template(
            message_dicts, tokenize=False, add_generation_prompt=True
        )

    model._to_chat_prompt = _patched_to_chat_prompt

    def _patched_generate(messages, stop=None, run_manager=None, **kwargs):
        messages = _inject_tool_descriptions(messages, tools)
        result = original_generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return _parse_text_tool_calls(result, valid_tool_names)

    model._generate = _patched_generate

    return model


def _ensure_sync_tools(tools: List) -> List:
    out = []
    for t in tools:
        if isinstance(t, StructuredTool) and t.func is None and t.coroutine is not None:
            t.func = lambda *args, _coro=t.coroutine, **kwargs: asyncio.run(_coro(*args, **kwargs))
        out.append(t)
    return out


class SimpleReActAgent:
    def __init__(self, llm, tools, prompt, max_iterations=10):
        sync_tools = _ensure_sync_tools(list(tools))
        wrapped = wrap_model_for_tools(llm, sync_tools)
        self._agent = create_agent(
            model=wrapped,
            tools=sync_tools,
            system_prompt=prompt,
        )
        self.max_iterations = max_iterations

    def invoke(self, input_dict: dict) -> dict:
        return self._agent.invoke(
            input_dict,
            config={"recursion_limit": self.max_iterations * 2 + 5},
        )
