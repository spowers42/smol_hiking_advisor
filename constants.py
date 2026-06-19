APP_TITLE = "Smol Hiking Advisor"
APP_DESCRIPTION = "Ask about hiking trails in New Hampshire"
APP_THEME_COLOR = "teal"

SYSTEM_PROMPT = (
    "You are a knowledgeable hiking advisor for New Hampshire trails. "
    "Provide helpful, safe advice about hikes, conditions, and preparation."
)

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
