import json
import re
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from langchain_core.messages import ToolCall

_FN_CALL_RE = re.compile(
    r'\{\s*"function"\s*:\s*"([^"]+)"(?:\s*,\s*(?:"arguments"\s*:\s*)?(\{.*?\}|".*?"))?\s*\}',
    re.DOTALL,
)


@dataclass
class ParsedOutput:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


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


def parse_model_output(text: str, valid_tool_names: set[str]) -> ParsedOutput:
    parsed = parse_tool_call(text)
    if parsed and parsed[0] in valid_tool_names:
        name, args = parsed
        return ParsedOutput(
            text=text,
            tool_calls=[ToolCall(name=name, args=args, id=f"call_{uuid4().hex[:12]}")],
        )
    return ParsedOutput(text=text)
