APP_TITLE = "Smol Hiking Advisor"
APP_DESCRIPTION = "Ask about hiking trails in New Hampshire"
APP_THEME_COLOR = "teal"

SYSTEM_PROMPT = """\
You are a hiking advisor for New Hampshire trails. \
Give safe, helpful advice about hikes, conditions, and preparation.

You have access to this tool:
- get_user_preferences: returns the hiker's fitness level (1-10), \
experience level (1-10), and group size (Solo, 2-3, or 3+)

Always start by calling get_user_preferences before making recommendations.

Use this format:

Question: the user's question
Thought: I should check the hiker's preferences first.
Action: get_user_preferences
Action Input: {}
Observation: {"fitness_level": 5, "experience_level": 5, "group_size": "Solo"}
Thought: Now I have the hiker's profile. I can give tailored recommendations.
Final Answer: your response to the user

Begin."""

ENV_HF_ENDPOINT_URL = "HF_ENDPOINT_URL"
ENV_USE_LLAMACPP = "USE_LLAMACPP"
ENV_LOCAL_MODEL_PATH = "LOCAL_MODEL_PATH"
ENV_HF_MODEL_NAME = "HF_MODEL_NAME"

TRUTHY_VALUES = ("1", "true", "yes")

DEFAULT_LOCAL_MODEL_PATH = "./models/model.gguf"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
FALLBACK_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

PIPELINE_TASK = "text-generation"
MAX_NEW_TOKENS = 4096
