import asyncio
import re
from typing import Any, List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langchain_core.tools.render import render_text_description_and_args
from langchain_huggingface import ChatHuggingFace
from langchain_openai import ChatOpenAI

from app.response_parser import parse_model_output


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
        parsed = parse_model_output(gen.message.content, valid_tool_names)
        if parsed.tool_calls:
            msg = AIMessage(
                content=parsed.text,
                tool_calls=parsed.tool_calls,
            )
            generations.append(ChatGeneration(message=msg))
        else:
            generations.append(gen)
    return ChatResult(generations=generations, llm_output=result.llm_output)


_TOOL_CALL_JSON_RE = re.compile(r'\{\s*"(?:name|function)"\s*:')


def _strip_tool_call_content(result: ChatResult) -> ChatResult:
    """Strip tool-call JSON from AIMessage content when tool_calls are present.

    Some model servers (e.g. llama.cpp) include the raw generated tool-call
    text in *both* the ``content`` field and the structured ``tool_calls``
    field of assistant messages.  When that content is sent back to the server
    on the next request the server's chat-template parser chokes on it.

    This function detects that situation and clears the content.
    """
    generations = []
    for gen in result.generations:
        msg = gen.message
        if (
            isinstance(msg, AIMessage)
            and msg.tool_calls
            and isinstance(msg.content, str)
            and msg.content
            and _TOOL_CALL_JSON_RE.search(msg.content)
        ):
            msg.content = ""
        generations.append(gen)
    return ChatResult(generations=generations, llm_output=result.llm_output)


def _patch_openai_content_stripping(model: ChatOpenAI) -> None:
    """Patch ``_generate`` so assistant content is stripped when tool_calls exist."""
    original_generate = model._generate

    def _patched_generate(messages, stop=None, run_manager=None, **kwargs):
        result = original_generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return _strip_tool_call_content(result)

    model._generate = _patched_generate


def _needs_local_tool_handling(llm: Any) -> bool:
    if not isinstance(llm, ChatHuggingFace):
        return False
    from langchain_huggingface.chat_models.huggingface import (
        _is_huggingface_endpoint,
        _is_huggingface_textgen_inference,
    )

    return not (_is_huggingface_endpoint(llm.llm) or _is_huggingface_textgen_inference(llm.llm))


def wrap_model_for_tools(model: Any, tools: List) -> Any:
    if not tools:
        return model

    if isinstance(model, ChatOpenAI):
        _patch_openai_content_stripping(model)
        return model

    if not _needs_local_tool_handling(model):
        return model

    valid_tool_names = {t.name for t in tools}
    original_generate = model._generate
    original_to_chatml = model._to_chatml_format

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


def _patch_empty_schema(tools: List) -> None:
    """Work around a llama.cpp server bug that rejects empty tool parameter schemas.

    The llama.cpp server returns HTTP 500 when it receives a tool definition with
    an empty parameters object::

        "parameters": {"type": "object", "properties": {}}

    This happens at tool *registration* time, before any tool is called. The error
    originates inside the server's grammar/conversion layer which expects *at least
    one* property when ``type`` is ``"object"``.

    We work around it by injecting a hidden ``unused`` string field with a default
    value of ``""`` into every empty schema. Both Pyright model schemas and raw
    dict schemas are patched.

    The corresponding tool functions should also accept ``**kwargs`` so they don't
    break if the model ever passes the ``unused`` parameter.
    """
    from pydantic import BaseModel, Field

    for t in tools:
        schema = t.args_schema
        if not schema:
            continue
        if isinstance(schema, type) and issubclass(schema, BaseModel) and not schema.model_fields:

            class _Patched(schema):  # type: ignore
                unused: str = Field(default="", description="(internal)")

            _Patched.__name__ = schema.__name__
            t.args_schema = _Patched
        elif isinstance(schema, dict):
            props = schema.get("properties", {})
            if not props:
                schema.setdefault("properties", {})["unused"] = {
                    "type": "string",
                    "description": "(internal)",
                }


class SimpleReActAgent:
    def __init__(self, llm, tools, prompt, max_iterations=10):
        sync_tools = _ensure_sync_tools(list(tools))
        _patch_empty_schema(sync_tools)
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
