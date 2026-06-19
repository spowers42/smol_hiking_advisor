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

```bash
task run
```

Other [Task](https://taskfile.dev/) commands:

```bash
task lint     # ruff check
task test     # pytest
task all      # lint + test
```

## Technical Details

- [Mt Washington MCP server](https://github.com/spowers42/mt_washington_obs_mcp)
- [Finetuning dataset](https://huggingface.co/datasets/scott-ml/nh_hiking_recomendation)
- Finetuned Model (WIP)



