import json
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_huggingface import ChatHuggingFace
from pydantic import PrivateAttr

_FN_CALL_RE = re.compile(
    r'\{\s*"function"\s*:\s*"([^"]+)"(?:\s*,\s*(?:"arguments"\s*:\s*)?(\{.*?\}|".*?"))?\s*\}',
    re.DOTALL,
)


def _tools_block(tools: List[Dict[str, Any]]) -> str:
    parts = ["You have access to the following functions. Use them when needed."]
    for t in tools:
        name = t.get("name", "?")
        desc = t.get("description", "")
        parts.append(f"\nFunction: {name}")
        parts.append(f"Description: {desc}")
        params = t.get("parameters", {})
        if params:
            parts.append(f"Parameters: {json.dumps(params)}")
    parts.append(
        "\n\nWhen you need to call a function, respond with ONLY a JSON object "
        'on its own line in this format:\n{"function": "name", "arguments": {}}\n'
        "Call one function at a time."
    )
    return "\n".join(parts)


class ToolCallingChatModel(BaseChatModel):
    model_config = {"arbitrary_types_allowed": True}
    _chat: Any = PrivateAttr()

    def __init__(self, chat: ChatHuggingFace, **kwargs):
        super().__init__(**kwargs)
        self._chat = chat

    @property
    def _llm_type(self) -> str:
        return "tool-calling-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.pop("tools", None)
        if tools:
            messages = self._inject_tool_descriptions(messages, tools)

        message_dicts = [self._to_chatml_format(m) for m in messages]
        prompt = self._chat.tokenizer.apply_chat_template(
            message_dicts, tokenize=False, add_generation_prompt=True
        )

        llm_result = self._chat.llm._generate(
            prompts=[prompt],
            stop=stop,
            run_manager=run_manager,
            skip_prompt=True,
            **kwargs,
        )

        generations = []
        for gen in llm_result.generations[0]:
            text = gen.text.strip()
            match = _FN_CALL_RE.search(text)
            if match:
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
                msg = AIMessage(
                    content=text,
                    tool_calls=[
                        ToolCall(
                            name=name,
                            args=args,
                            id=f"call_{uuid4().hex[:12]}",
                        )
                    ],
                )
            else:
                msg = AIMessage(content=text)
            generations.append(ChatGeneration(message=msg))

        return ChatResult(generations=generations, llm_output=llm_result.llm_output)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _inject_tool_descriptions(
        self, messages: List[BaseMessage], tools: List[Dict[str, Any]]
    ) -> List[BaseMessage]:
        messages = list(messages)
        block = _tools_block(tools)
        for i, m in enumerate(messages):
            if isinstance(m, SystemMessage):
                messages[i] = SystemMessage(content=m.content + "\n\n" + block)
                return messages
        messages.insert(0, SystemMessage(content=block))
        return messages

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, AIMessage):
            return {"role": "assistant", "content": message.content}
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        if isinstance(message, ToolMessage):
            return {
                "role": "user",
                "content": f"Tool result ({message.name}): {message.content}",
            }
        raise ValueError(f"Unknown message type: {type(message)}")
