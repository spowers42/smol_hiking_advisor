import json
import re
from typing import Optional, Tuple

_FN_CALL_RE = re.compile(
    r'\{\s*"function"\s*:\s*"([^"]+)"(?:\s*,\s*(?:"arguments"\s*:\s*)?(\{.*?\}|".*?"))?\s*\}',
    re.DOTALL,
)


def parse_tool_call(text: str) -> Optional[Tuple[str, dict]]:
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
