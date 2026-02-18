"""
MCP Client for AgentCore Gateway connection.

Provides MCP client for communicating with AgentCore Gateway,
which routes tool calls to Lambda functions.
"""

from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger

from src.config import settings


def get_mcp_client() -> MultiServerMCPClient:
    """
    Create and return an MCP Client for AgentCore Gateway.

    Returns:
        MultiServerMCPClient configured for AgentCore Gateway.

    Raises:
        RuntimeError: If GATEWAY_URL is not set.
    """
    gateway_url = settings.GATEWAY_URL
    if not gateway_url:
        raise RuntimeError("Missing required configuration: GATEWAY_URL")

    logger.debug(f"Creating MCP client for gateway: {gateway_url}")

    return MultiServerMCPClient(
        {
            "agentcore_gateway": {
                "transport": "streamable_http",
                "url": gateway_url,
            }
        }
    )


def is_mcp_configured() -> bool:
    """
    Check if MCP Gateway is configured.

    Returns:
        True if GATEWAY_URL is set.
    """
    return bool(settings.GATEWAY_URL)
