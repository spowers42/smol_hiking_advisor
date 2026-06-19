import os
import warnings

from dotenv import load_dotenv

import constants

warnings.filterwarnings(
    "ignore",
    message=".*generation_config together with generation-related arguments.*",
)

load_dotenv()


def get_llm():
    hf_endpoint_url = os.environ.get(constants.ENV_HF_ENDPOINT_URL)
    if hf_endpoint_url:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        llm = HuggingFaceEndpoint(
            endpoint_url=hf_endpoint_url,
            task=constants.PIPELINE_TASK,
        )
        return ChatHuggingFace(llm=llm)

    use_llamacpp = os.environ.get(constants.ENV_USE_LLAMACPP, "").lower() in constants.TRUTHY_VALUES
    if use_llamacpp:
        from langchain_community.chat_models import ChatLlamaCpp

        model_path = os.environ.get(
            constants.ENV_LOCAL_MODEL_PATH, constants.DEFAULT_LOCAL_MODEL_PATH
        )
        return ChatLlamaCpp(model_path=model_path)

    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    model_name = os.environ.get(constants.ENV_HF_MODEL_NAME, constants.DEFAULT_MODEL_NAME)
    fallback_model = constants.FALLBACK_MODEL_NAME
    models_to_try = [model_name] if model_name == fallback_model else [model_name, fallback_model]
    for attempt in models_to_try:
        try:
            llm = HuggingFacePipeline.from_model_id(
                model_id=attempt,
                task=constants.PIPELINE_TASK,
                pipeline_kwargs={"max_new_tokens": constants.MAX_NEW_TOKENS},
            )
            break
        except OSError:
            if attempt == models_to_try[-1]:
                raise
    return ChatHuggingFace(llm=llm)
