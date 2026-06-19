import os
from unittest.mock import MagicMock, patch

from app.config import get_llm


class TestGetLLM:
    @patch("langchain_huggingface.ChatHuggingFace")
    @patch("langchain_huggingface.HuggingFaceEndpoint")
    def test_uses_hf_endpoint_when_env_var_set(self, mock_hf_endpoint, mock_chat_hf):
        mock_chat_hf.return_value = MagicMock()
        with patch.dict(os.environ, {"HF_ENDPOINT_URL": "https://test.hf.space"}):
            result = get_llm()
        mock_hf_endpoint.assert_called_once_with(
            endpoint_url="https://test.hf.space",
            task="text-generation",
        )
        mock_chat_hf.assert_called_once_with(llm=mock_hf_endpoint.return_value)
        assert result is mock_chat_hf.return_value

    @patch("langchain_community.chat_models.ChatLlamaCpp")
    def test_uses_llamacpp_when_flag_set(self, mock_chat_llama):
        mock_chat_llama.return_value = MagicMock()
        with patch.dict(
            os.environ,
            {"USE_LLAMACPP": "true", "LOCAL_MODEL_PATH": "./models/model.gguf"},
            clear=True,
        ):
            result = get_llm()
        mock_chat_llama.assert_called_once_with(model_path="./models/model.gguf")
        assert result is mock_chat_llama.return_value

    @patch("langchain_community.chat_models.ChatLlamaCpp")
    def test_uses_llamacpp_with_custom_path(self, mock_chat_llama):
        mock_chat_llama.return_value = MagicMock()
        with patch.dict(
            os.environ,
            {
                "USE_LLAMACPP": "true",
                "LOCAL_MODEL_PATH": "/custom/path/model.gguf",
            },
            clear=True,
        ):
            result = get_llm()
        mock_chat_llama.assert_called_once_with(model_path="/custom/path/model.gguf")
        assert result is mock_chat_llama.return_value

    @patch("langchain_huggingface.ChatHuggingFace")
    @patch("langchain_huggingface.HuggingFacePipeline")
    def test_uses_transformers_by_default(self, mock_hf_pipeline, mock_chat_hf):
        mock_chat_hf.return_value = MagicMock()
        with patch.dict(os.environ, {}, clear=True):
            result = get_llm()
        mock_hf_pipeline.from_model_id.assert_called_once_with(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 4096},
        )
        mock_chat_hf.assert_called_once_with(llm=mock_hf_pipeline.from_model_id.return_value)
        assert result is mock_chat_hf.return_value

    @patch("langchain_huggingface.ChatHuggingFace")
    @patch("langchain_huggingface.HuggingFacePipeline")
    def test_uses_transformers_with_custom_model_name(self, mock_hf_pipeline, mock_chat_hf):
        mock_chat_hf.return_value = MagicMock()
        with patch.dict(
            os.environ,
            {"HF_MODEL_NAME": "facebook/opt-125m"},
            clear=True,
        ):
            result = get_llm()
        mock_hf_pipeline.from_model_id.assert_called_once_with(
            model_id="facebook/opt-125m",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 4096},
        )
        assert result is mock_chat_hf.return_value

    @patch("langchain_huggingface.ChatHuggingFace")
    @patch("langchain_huggingface.HuggingFacePipeline")
    def test_falls_back_to_phi_on_oserror(self, mock_hf_pipeline, mock_chat_hf):
        mock_chat_hf.return_value = MagicMock()
        pipeline_mock = MagicMock()
        mock_hf_pipeline.from_model_id.side_effect = [
            OSError("403 Client Error"),
            pipeline_mock,
        ]
        with patch.dict(os.environ, {}, clear=True):
            result = get_llm()
        calls = mock_hf_pipeline.from_model_id.call_args_list
        assert calls[0][1]["model_id"] == "meta-llama/Llama-3.2-3B-Instruct"
        assert calls[1][1]["model_id"] == "microsoft/Phi-3-mini-4k-instruct"
        mock_chat_hf.assert_called_once_with(llm=pipeline_mock)
        assert result is mock_chat_hf.return_value
