# smol_hiking_advisor

This is a specialist hiking planning agent for trails in New Hampshire, specifically in the White Mountains.  It uses a fine tuned (WIP) version of Llama3 8B to help plan hikes and progression plans for hikers of all levels.  It also utilizes the Mt. Washington Observatory MCP server to get weather forecast data and real time conditions.  

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (package manager)

## Setup

```bash
git clone <repo-url>
cd smol_hiking_advisor
uv sync --extra dev
```

Copy and configure environment variables:

```bash
cp .env.example .env
```

Three LLM backends are supported (evaluated in priority order):

| Variable | Purpose |
|---|---|
| `HF_ENDPOINT_URL` | Hugging Face Inference Endpoint (highest priority) |
| `USE_LLAMACPP` | Enable local GGUF model via `llama-cpp-python` |
| `LOCAL_MODEL_PATH` | Path to the GGUF model file (used with `USE_LLAMACPP`) |
| `HF_MODEL_NAME` | Hugging Face model name for local transformers (lowest priority) |

## Running

Start the llama.cpp server with multi-model support:

```bash
llama-server --models-preset models.ini --host 127.0.0.1 --port 8080
```

The server auto-downloads models referenced in `models.ini` (via `hf` keys) on first load. To pre-download them:

```bash
task fetch-models
```

Then run the app:

```bash
task run
```

Other [Task](https://taskfile.dev/) commands:

```bash
task lint     # ruff check
task test     # pytest
task all      # lint + test
```

## Models

`models.ini` at the project root configures models in [llama.cpp preset format](https://github.com/ggml-org/llama.cpp/blob/master/docs/preset.md). Each section other than `[*]` defines a model alias that is exposed via the [router server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md):

```ini
[*]
n-gpu-layers = 999
flash-attn   = 1
no-mmap      = true

[llama-3.1-8b]
hf = unsloth/Llama-3.1-8B-Instruct-GGUF:UD-Q4_K_XL
```

The server also supports `--models-dir models/` to look for local `.gguf` files alongside the HF auto-download.

To pre-download all model files into `models/` (gitignored):

```bash
task fetch-models
```

### DeepEval Judge LLM

`app/models/judge_llm.py` provides a `JudgeLLM` class that wraps the running llama.cpp server as a custom DeepEval evaluation model. It reads `LLAMACPP_HOST` and `LLAMACPP_PORT` from the environment (same settings the agent uses). Use it with any DeepEval metric:

```python
from app.models.judge_llm import JudgeLLM
from deepeval.metrics import AnswerRelevancyMetric

judge = JudgeLLM()
metric = AnswerRelevancyMetric(model=judge)
```

## Technical Details

- [Mt Washington MCP server](https://github.com/spowers42/mt_washington_obs_mcp)
- [Finetuning dataset](https://huggingface.co/datasets/scott-ml/nh_hiking_recomendation)
- Finetuned Model (WIP)



