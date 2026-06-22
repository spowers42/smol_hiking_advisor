from typing import Any, List, Optional
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
from langchain_core.tools.render import render_text_description_and_args
from langchain_huggingface import ChatHuggingFace
from pydantic import PrivateAttr

from app.models.tool_serialization import parse_tool_call


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
            parsed = parse_tool_call(text)
            if parsed:
                name, args = parsed
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
        self, messages: List[BaseMessage], tools: List
    ) -> List[BaseMessage]:
        messages = list(messages)
        rendered = render_text_description_and_args(tools)
        instructions = (
            "\n\nWhen you need to call a function, respond with ONLY a JSON object "
            'on its own line in this format:\n{"function": "name", "arguments": {}}\n'
            "Call one function at a time."
        )
        block = rendered + instructions
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
