# Domain context

## Architecture

### AgentRuntime (`app/runtime.py`)

The agent's lifecycle and invocation seam. Accepts a `RuntimeConfig` that bundles the LLM, tools, prompt, and max iteration count. Callers wire their own dependencies — no global singletons, no side effects at import time.

- **RuntimeConfig** — data class holding `llm`, `tools`, `prompt`, `max_iterations`. The single point of configuration for an agent instance.
- **AgentRuntime** — wraps `SimpleReActAgent` behind a stable interface. Currently exposes `invoke(input_dict)`. Future depth: logging, tracing, retry logic.

### SimpleReActAgent (`app/agent.py`)

The ReAct loop. Not meant to be used directly by production code — go through `AgentRuntime`.

## Upstream data sources

### Mt Washington Observatory (MCP)

Real-time weather conditions, summit forecasts, valley forecasts, past-24h statistics, and almanac data. Accessed through a FastMCP server at `mt-washington-weather.fastmcp.app`. Connection is managed by the weather tool module (`app/tools/weather.py`) and tools are injected into the runtime by the caller.

### Hiker profile

Per-session user preferences (fitness level, experience, group size). Stored in a module-level dict in `app/tools/user_preferences.py`. Set via the Gradio sidebar, read by the agent.
