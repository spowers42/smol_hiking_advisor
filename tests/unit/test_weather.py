from unittest.mock import AsyncMock, patch

import pytest

import app.tools.weather
from app.tools.weather import (
    connect_async,
    get_weather_tools,
    is_connected,
)


class TestGetWeatherTools:
    def test_returns_list_copy(self):
        tools = get_weather_tools()
        assert isinstance(tools, list)


class TestIsConnected:
    @pytest.fixture(autouse=True)
    def _reset_weather_state(self):
        app.tools.weather._mcp_client = None
        app.tools.weather._weather_tools = []

    def test_returns_false_when_not_connected(self):
        assert is_connected() is False


class TestConnectAsync:
    @pytest.fixture(autouse=True)
    def _reset_weather_state(self):
        app.tools.weather._mcp_client = None
        app.tools.weather._weather_tools = []

    async def test_skips_connection_when_url_empty(self, monkeypatch):
        monkeypatch.setenv("MCP_WEATHER_URL", "")
        await connect_async()
        assert len(get_weather_tools()) == 0

    async def test_handles_connection_error_gracefully(self):
        with (
            patch("app.tools.weather.MultiServerMCPClient") as mock_client_cls,
            patch("app.tools.weather._build_resource_tool") as mock_build,
        ):
            mock_client = AsyncMock()
            mock_client.get_tools.side_effect = ConnectionError("MCP server down")
            mock_client_cls.return_value = mock_client

            await connect_async()

            tools = get_weather_tools()
            assert len(tools) == 0
            assert app.tools.weather._mcp_client is None
            mock_build.assert_not_called()

    async def test_stores_mcp_and_resource_tools_on_success(self):
        fake_mcp_tool = AsyncMock()
        fake_mcp_tool.name = "extract_f6_csv"
        fake_mcp_tool.description = "Extract F6 data"

        fake_resource_tool = AsyncMock()
        fake_resource_tool.name = "get_current_conditions"
        fake_resource_tool.description = "Current summit weather"

        with (
            patch("app.tools.weather.MultiServerMCPClient") as mock_client_cls,
            patch(
                "app.tools.weather._build_resource_tool",
                return_value=fake_resource_tool,
            ) as mock_build,
        ):
            mock_client = AsyncMock()
            mock_client.get_tools.return_value = [fake_mcp_tool]
            mock_client_cls.return_value = mock_client

            await connect_async()

            tools = get_weather_tools()
            # 1 MCP tool + 5 resource tools
            assert len(tools) == 6
            assert tools[0].name == "extract_f6_csv"
            assert tools[1].name == "get_current_conditions"
            assert mock_build.call_count == 5


class TestResourceTool:
    @pytest.fixture(autouse=True)
    def _reset_weather_state(self):
        app.tools.weather._mcp_client = None
        app.tools.weather._weather_tools = []

    async def test_returns_graceful_message_on_session_failure(self):
        mock_client = AsyncMock(spec=["session"])
        mock_cm = AsyncMock()
        mock_cm.__aenter__.side_effect = ConnectionError("MCP server unreachable")
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_client.session.return_value = mock_cm
        app.tools.weather._mcp_client = mock_client

        tool = await app.tools.weather._build_resource_tool(
            "weather://current",
            "get_current_conditions",
            "Current summit weather",
        )

        result = await tool.ainvoke({})

        assert result == "Weather data is currently unavailable. Please try again later."
