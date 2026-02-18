"""
Browser infrastructure for AgentCore Browser integration.

This module provides a managed browser toolkit using AWS Bedrock AgentCore Browser,
which enables agents to interact with web applications in a secure, isolated environment.

Thread-based session isolation: Each unique thread_id in the config creates a
separate browser session, allowing concurrent operations.
"""

from typing import Dict, List

from langchain_aws.tools import create_browser_toolkit
from langchain_aws.tools.browser_toolkit import BrowserToolkit
from langchain_core.tools import BaseTool
from loguru import logger

from src.config import settings

# Global browser toolkit instance (singleton)
# The toolkit internally manages separate sessions per thread_id
_browser_toolkit: BrowserToolkit | None = None
_browser_tools: List[BaseTool] | None = None
_browser_tools_by_name: Dict[str, BaseTool] | None = None


def get_browser_toolkit() -> BrowserToolkit:
    """
    Get or create the browser toolkit singleton.

    The toolkit manages separate browser sessions for each thread_id passed
    via config when invoking tools or agents.

    Returns:
        BrowserToolkit: The initialized browser toolkit instance.
    """
    global _browser_toolkit, _browser_tools, _browser_tools_by_name

    if _browser_toolkit is None:
        region = settings.AWS_REGION
        _browser_toolkit, _browser_tools = create_browser_toolkit(region=region)
        _browser_tools_by_name = _browser_toolkit.get_tools_by_name()
        logger.info(f"Browser toolkit initialized for region: {region}")

    return _browser_toolkit


def get_browser_tools() -> List[BaseTool]:
    """
    Get the list of browser tools from the toolkit.

    Returns:
        List[BaseTool]: List of LangChain browser tools.
    """
    global _browser_tools
    get_browser_toolkit()  # Ensure toolkit is initialized
    return _browser_tools


def get_browser_tools_with_config(thread_id: str) -> List[BaseTool]:
    """
    Get browser tools with config pre-bound for a specific thread_id.

    This ensures the thread_id is always passed to each tool call,
    solving the issue where agents don't propagate config to tools.

    Args:
        thread_id: Unique identifier for the browser session.

    Returns:
        List[BaseTool]: List of browser tools with config bound.
    """
    get_browser_toolkit()  # Ensure toolkit is initialized
    config = {"configurable": {"thread_id": thread_id}}

    # Bind the config to each tool so thread_id is always used
    bound_tools = [tool.bind(config) for tool in _browser_tools]
    logger.info(f"Browser tools bound with thread_id: {thread_id}")
    return bound_tools


def get_browser_tools_by_name() -> Dict[str, BaseTool]:
    """
    Get browser tools as a dictionary keyed by tool name.

    This allows direct tool invocation with config:
        tools_by_name["navigate_browser"].invoke({"url": "..."}, config=config)

    Returns:
        Dict[str, BaseTool]: Dictionary of tool_name -> BaseTool.
    """
    global _browser_tools_by_name
    get_browser_toolkit()  # Ensure toolkit is initialized
    return _browser_tools_by_name


async def cleanup_browser_sessions() -> None:
    """
    Clean up all browser sessions and reset the toolkit singleton.

    Call this after each browser search to release resources and
    ensure fresh sessions for subsequent searches.
    """
    global _browser_toolkit, _browser_tools, _browser_tools_by_name

    if _browser_toolkit is not None:
        try:
            await _browser_toolkit.cleanup()
            logger.info("Browser sessions cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error cleaning up browser sessions: {e}")
        finally:
            # Reset singleton so next search gets a fresh toolkit
            _browser_toolkit = None
            _browser_tools = None
            _browser_tools_by_name = None
