# smol_hiking_advisor

This is a specialist hiking planning agent for trails in New Hampshire, specifically in the White Mountains.  It uses a fine tuned (WIP) version of Llama3 8B to help plan hikes and progression plans for hikers of all levels.  It also utilizes the Mt. Washington Observatory MCP server to get weather forecast data and real time conditions.  

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [task](https://taskfile.dev/) ( task runner)

## Setup

```bash
git clone <repo-url>
cd smol_hiking_advisor
task init
```
The init command will copy the sample environment configuration to .env and install all 
dependencies, including developer dependencies.  It also fetches the models used for the local llama.cpp server.

Configure environment variables:

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

All available [Task](https://taskfile.dev/) commands:

| Command | Description |
|---|---|
| `task init` | Full project setup (copies `.env`, installs deps, downloads models) |
| `task install` | Install project with dev dependencies |
| `task lint` | Run ruff linter and formatter checks |
| `task lint-fix` | Auto-fix lint issues and format |
| `task test` | Run unit tests (excludes evals) |
| `task eval` | Run agent evaluation suite (golden test cases) |
| `task run` | Run the Gradio app locally |
| `task build` | Build the Python package |
| `task publish` | Publish to PyPI |
| `task fetch-models` | Download GGUF model files from `models.ini` |
| `task all` | Run lint + test + eval |

## Models

`models.ini` at the project root configures models in [llama.cpp preset format](https://github.com/ggml-org/llama.cpp/blob/master/docs/preset.md). 

```
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



