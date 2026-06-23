APP_TITLE = "Smol Hiking Advisor"
APP_DESCRIPTION = "Ask about hiking trails in New Hampshire"
APP_THEME_COLOR = "teal"

SYSTEM_PROMPT_TEMPLATE = """\
You are a hiking advisor for New Hampshire trails. \
Give safe, helpful advice about hikes, conditions, and preparation.

Always start by calling get_user_preferences before making \
recommendations.

You have access to load_skill to load specialized expertise \
on demand. Use it when a question needs deep domain knowledge.

{skills_section}

For weather questions, call BOTH get_current_conditions \
AND get_summit_forecast — one at a time. First call \
get_current_conditions, wait for the result, then call \
get_summit_forecast.

Never make up data. Only use what the tools return."""


def build_system_prompt() -> str:
    from app.skills import build_skills_section

    skills_section = build_skills_section()
    return SYSTEM_PROMPT_TEMPLATE.format(skills_section=skills_section)


ENV_HF_ENDPOINT_URL = "HF_ENDPOINT_URL"
ENV_USE_LLAMACPP = "USE_LLAMACPP"
ENV_LLAMACPP_HOST = "LLAMACPP_HOST"
ENV_LLAMACPP_PORT = "LLAMACPP_PORT"
ENV_LLAMACPP_MODEL_NAME = "LLAMACPP_MODEL_NAME"
ENV_HF_MODEL_NAME = "HF_MODEL_NAME"

ENV_MCP_WEATHER_URL = "MCP_WEATHER_URL"
ENV_MCP_WEATHER_API_KEY = "MCP_WEATHER_API_KEY"

MCP_WEATHER_URL = "https://mt-washington-weather.fastmcp.app/mcp"

TRUTHY_VALUES = ("1", "true", "yes")

DEFAULT_LLAMACPP_HOST = "localhost"
DEFAULT_LLAMACPP_PORT = 8080
DEFAULT_LLAMACPP_MODEL = "default"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
FALLBACK_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

PIPELINE_TASK = "text-generation"
MAX_NEW_TOKENS = 4096
