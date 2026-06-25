import asyncio
import logging
import os

from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.resources import load_mcp_resources

from app import constants

logger = logging.getLogger(__name__)

_mcp_client: MultiServerMCPClient | None = None
_weather_tools: list = []

RESOURCE_TOOLS = [
    (
        "weather://current",
        "get_current_conditions",
        "Get the current summit weather conditions on Mt Washington "
        "including temperature, wind speed, wind gusts, wind direction, and METAR data",
    ),
    (
        "weather://outlook/summit",
        "get_summit_forecast",
        "Get the Higher Summits Forecast for Mt Washington with 4-period discussion",
    ),
    (
        "weather://outlook/valley",
        "get_valley_forecast",
        "Get the Valley Forecast for the Mt Washington area with 4-period discussion",
    ),
    (
        "weather://outlook/statistics",
        "get_weather_statistics",
        "Get the past 24 hours weather statistics for Mt Washington "
        "including max/min temperature, peak wind gust, precipitation, and snowfall",
    ),
    (
        "weather://outlook/almanac",
        "get_weather_almanac",
        "Get the weather almanac for Mt Washington "
        "including records, monthly averages, sunrise and sunset times",
    ),
]


def get_weather_tools() -> list:
    return list(_weather_tools)


def is_connected() -> bool:
    return _mcp_client is not None


async def _build_resource_tool(uri: str, name: str, description: str):
    client = _mcp_client

    @tool(description=description)
    async def resource_tool(**kwargs) -> str:
        try:
            async with client.session("mt-washington") as session:
                blobs = await load_mcp_resources(session, uris=[uri])
        except Exception:
            logger.warning("MCP resource %s unavailable", uri)
            return "Weather data is currently unavailable. Please try again later."
        if not blobs:
            return "No data available."
        parts = []
        for b in blobs:
            raw = b.as_bytes() or b.as_string()
            if raw is not None:
                raw = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                parts.append(raw)
        return "\n".join(parts)

    resource_tool.name = name
    return resource_tool


async def connect_async() -> None:
    global _mcp_client, _weather_tools

    url = os.environ.get(constants.ENV_MCP_WEATHER_URL, constants.MCP_WEATHER_URL)
    if not url:
        logger.info("No MCP weather URL configured; skipping connection")
        return

    headers = None
    api_key = os.environ.get(constants.ENV_MCP_WEATHER_API_KEY)
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}

    try:
        conn = {
            "transport": "streamable_http",
            "url": url,
            "headers": headers,
        }
        client = MultiServerMCPClient({"mt-washington": conn})
        _mcp_client = client

        mcp_tools = await client.get_tools()

        resource_tools = await asyncio.gather(
            *[_build_resource_tool(u, n, d) for u, n, d in RESOURCE_TOOLS]
        )

        _weather_tools = [*mcp_tools, *resource_tools]
        logger.info(
            "Connected to Mt Washington MCP server, got %d MCP tool(s) and %d resource tool(s)",
            len(mcp_tools),
            len(resource_tools),
        )
    except Exception as exc:
        logger.warning(
            "Failed to connect to Mt Washington MCP server (%s); continuing without weather tools",
            exc,
        )
        _mcp_client = None
        _weather_tools = []


def connect() -> None:
    try:
        asyncio.get_running_loop()
        logger.warning("Called from async context; connect_async() must be awaited separately")
    except RuntimeError:
        asyncio.run(connect_async())
