import json
import os

from deepeval.models import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel


class JudgeLLM(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: str = "llama.cpp",
        host: str | None = None,
        port: int | None = None,
    ):
        host = host or os.environ.get("LLAMACPP_HOST", "localhost")
        port = port or int(os.environ.get("LLAMACPP_PORT", "8080"))
        base_url = f"http://{host}:{port}/v1"
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.async_client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
        self.model_name = model_name

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: BaseModel | None = None) -> str | BaseModel:
        messages = [{"role": "user", "content": prompt}]
        if schema is not None:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "You must respond with valid JSON matching the expected schema.",
                },
            )
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }
        else:
            kwargs = {"model": self.model_name, "messages": messages}
        resp = self.client.chat.completions.create(**kwargs)
        output = resp.choices[0].message.content
        if schema is not None:
            return schema(**json.loads(output))
        return output

    async def a_generate(self, prompt: str, schema: BaseModel | None = None) -> str | BaseModel:
        messages = [{"role": "user", "content": prompt}]
        if schema is not None:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "You must respond with valid JSON matching the expected schema.",
                },
            )
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }
        else:
            kwargs = {"model": self.model_name, "messages": messages}
        resp = await self.async_client.chat.completions.create(**kwargs)
        output = resp.choices[0].message.content
        if schema is not None:
            return schema(**json.loads(output))
        return output

    def get_model_name(self):
        return self.model_name
