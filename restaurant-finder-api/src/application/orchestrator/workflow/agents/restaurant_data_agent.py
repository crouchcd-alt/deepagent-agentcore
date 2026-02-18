"""
Restaurant Data Agent - MCP-based restaurant search via AgentCore Gateway.

This agent uses the MCP (Model Context Protocol) client to communicate with
an AgentCore Gateway, which routes requests to a Lambda function that provides
restaurant data.

Architecture:
    Agent → MCP Client → AgentCore Gateway → Lambda Function → Restaurant Data

The Lambda function currently returns dummy data for development/testing,
but can be connected to a real database or external API in production.
"""

import json
from typing import Any

from loguru import logger

from src.domain.models import Restaurant, RestaurantSearchResult, PriceRange, PRICE_RANGE_MAP
from src.infrastructure.mcp_client import get_mcp_client, is_mcp_configured


# =============================================================================
# Constants
# =============================================================================

# Tool name as defined in the AgentCore Gateway
SEARCH_RESTAURANTS_TOOL = "search_restaurants"


# =============================================================================
# MCP Tool Execution
# =============================================================================

async def call_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Call a tool via the MCP client connected to AgentCore Gateway.

    Args:
        tool_name: Name of the tool to invoke (e.g., "search_restaurants").
        arguments: Tool-specific arguments.

    Returns:
        Tool execution result as a dictionary.

    Raises:
        RuntimeError: If MCP client fails or tool execution errors.
    """
    try:
        # As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient is no longer a context manager
        client = get_mcp_client()

        # Get available tools from the gateway (returns LangChain BaseTool objects)
        tools = await client.get_tools()

        # Find the matching tool (AgentCore prefixes tool names)
        target_tool = None
        for tool in tools:
            # Tool names come as "LambdaTarget___<tool_name>" from gateway
            if tool.name.endswith(f"___{tool_name}") or tool.name == tool_name:
                target_tool = tool
                break

        if not target_tool:
            available = [t.name for t in tools]
            logger.warning(f"Tool '{tool_name}' not found. Available: {available}")
            raise RuntimeError(f"Tool '{tool_name}' not found in gateway")

        logger.info(f"Invoking MCP tool: {target_tool.name}")

        # Invoke the tool using LangChain's ainvoke method
        # The tools returned by get_tools() are LangChain BaseTool objects
        result = await target_tool.ainvoke(arguments)

        # Log the raw result for debugging
        logger.info(f"MCP tool raw result type: {type(result).__name__}")

        # Parse result - MCP tools return different formats
        # Format 1: List with TextContent items [{'type': 'text', 'text': '...', 'id': '...'}]
        # Format 2: String (JSON)
        # Format 3: Dict

        if isinstance(result, list) and len(result) > 0:
            # Handle MCP TextContent format: [{'type': 'text', 'text': '...'}]
            first_item = result[0]
            if isinstance(first_item, dict) and first_item.get("type") == "text":
                text_content = first_item.get("text", "")
                logger.info(f"MCP tool result is TextContent list, extracting text field")
                try:
                    # Parse the outer JSON (may contain statusCode and body)
                    outer_parsed = json.loads(text_content)
                    logger.debug(f"Outer parsed keys: {list(outer_parsed.keys()) if isinstance(outer_parsed, dict) else 'not a dict'}")

                    # Check if it's a Lambda response with statusCode and body
                    if isinstance(outer_parsed, dict) and "body" in outer_parsed:
                        body = outer_parsed.get("body")
                        if isinstance(body, str):
                            # Body is a JSON string, parse it
                            inner_parsed = json.loads(body)
                            logger.info(f"MCP tool result parsed from Lambda body, type: {type(inner_parsed).__name__}")
                            return inner_parsed if isinstance(inner_parsed, dict) else {"result": inner_parsed}
                        else:
                            return body if isinstance(body, dict) else {"result": body}
                    else:
                        return outer_parsed if isinstance(outer_parsed, dict) else {"result": outer_parsed}
                except json.JSONDecodeError as e:
                    logger.warning(f"MCP tool TextContent is not valid JSON: {text_content[:500]}, error: {e}")
                    return {"result": text_content}
            else:
                logger.warning(f"MCP tool result is list but not TextContent format")
                return {"result": str(result)}

        elif isinstance(result, str):
            try:
                parsed = json.loads(result)
                logger.info(f"MCP tool result parsed from JSON string, type: {type(parsed).__name__}")
                return parsed if isinstance(parsed, dict) else {"result": parsed}
            except json.JSONDecodeError:
                logger.warning(f"MCP tool result is not valid JSON: {result[:500]}")
                return {"result": result}

        elif isinstance(result, dict):
            logger.info(f"MCP tool result is already a dict with keys: {list(result.keys())}")
            return result

        else:
            logger.warning(f"MCP tool result is unexpected type: {type(result).__name__}")
            return {"result": str(result)}

    except Exception as e:
        logger.error(f"MCP tool call failed: {e}")
        raise RuntimeError(f"Failed to call MCP tool '{tool_name}': {e}")


# =============================================================================
# Restaurant Parsing
# =============================================================================

def parse_restaurant(data: dict) -> Restaurant:
    """
    Parse a dictionary into a Restaurant model.

    Args:
        data: Dictionary containing restaurant data from Lambda/SearchAPI.

    Returns:
        Restaurant: Parsed restaurant object with defaults for missing fields.
    """
    price_str = data.get("price_range", "$$")
    price_range = PRICE_RANGE_MAP.get(price_str, PriceRange.MODERATE)

    # Handle rating - ensure it's a valid float
    rating = data.get("rating", 0.0)
    if isinstance(rating, str):
        try:
            rating = float(rating)
        except ValueError:
            rating = 0.0
    rating = float(rating) if rating else 0.0

    # Handle review count
    review_count = data.get("review_count", 0)
    if isinstance(review_count, str):
        review_count = int("".join(filter(str.isdigit, review_count)) or "0")
    review_count = int(review_count) if review_count else 0

    return Restaurant(
        name=data.get("name", "Unknown Restaurant"),
        cuisine_type=data.get("cuisine_type", "Various"),
        rating=rating,
        review_count=review_count,
        price_range=price_range,
        address=data.get("address", ""),
        city=data.get("city", ""),
        phone=data.get("phone", ""),
        website=data.get("website", ""),
        features=data.get("features", []),
        dietary_options=data.get("dietary_options", []),
        operating_hours=data.get("operating_hours", ""),
        reservation_available=data.get("reservation_available", False),
    )


def _convert_to_string_dict(params: dict) -> dict[str, str]:
    """Convert a dict with mixed types to dict[str, str] for search_filters."""
    result = {}
    for key, value in params.items():
        if isinstance(value, list):
            result[key] = ", ".join(str(v) for v in value) if value else ""
        else:
            result[key] = str(value)
    return result


def parse_search_result(
    response: Any,
    query: str,
    search_params: dict,
) -> RestaurantSearchResult:
    """
    Parse Lambda/SearchAPI response into RestaurantSearchResult.

    Args:
        response: Response from Lambda function (dict or other).
        query: Original search query.
        search_params: Search parameters used.

    Returns:
        RestaurantSearchResult with parsed restaurants.
    """
    restaurants = []

    # Convert search_params to string dict for the model
    search_filters = _convert_to_string_dict(search_params)

    # Log the response structure for debugging
    logger.debug(f"Parsing response of type: {type(response).__name__}")

    # Handle case where response is not a dict
    if not isinstance(response, dict):
        logger.warning(f"Response is not a dict: {type(response).__name__}, value: {str(response)[:500]}")
        return RestaurantSearchResult(
            query=query,
            total_results=0,
            restaurants=[],
            search_location=search_params.get("location", ""),
            search_filters=search_filters,
            data_source="searchapi",
            notes=f"Unexpected response format: {str(response)[:200]}",
        )

    # Handle nested result structure from Lambda
    result_data = response.get("result", response)

    # If result_data is still not a dict, try to handle it
    if not isinstance(result_data, dict):
        logger.warning(f"result_data is not a dict: {type(result_data).__name__}")
        result_data = {"restaurants": [], "message": str(result_data)}

    restaurant_list = result_data.get("restaurants", [])
    logger.info(f"Found {len(restaurant_list)} restaurants in response")

    for item in restaurant_list:
        if isinstance(item, dict):
            restaurants.append(parse_restaurant(item))

    total_found = result_data.get("total_found", len(restaurants))
    message = result_data.get("message", "")

    # Include search query used if available
    search_query_used = result_data.get("search_query_used", "")
    if search_query_used:
        message = f"{message} Search query: {search_query_used}"

    # Check for errors in response
    error = result_data.get("error", "")
    if error:
        message = f"{message} Error: {error}"

    return RestaurantSearchResult(
        query=query,
        total_results=total_found,
        restaurants=restaurants,
        search_location=search_params.get("location", ""),
        search_filters=search_filters,
        data_source="searchapi",
        notes=message.strip(),
    )


# =============================================================================
# Main Entry Point
# =============================================================================

async def run_restaurant_data_agent(
    query: str,
    cuisine: str | None = None,
    location: str | None = None,
    price_range: str | None = None,
    dietary_restrictions: list[str] | None = None,
    limit: int = 5,
) -> RestaurantSearchResult:
    """
    Search for restaurants via AgentCore Gateway MCP connection using SearchAPI.

    This agent connects to the AgentCore Gateway using MCP protocol,
    which routes the request to a Lambda function that performs web searches
    via SearchAPI to find real restaurant information.

    Args:
        query: Natural language search query for context.
            Examples:
            - "Italian restaurants in San Francisco"
            - "Vegetarian Thai food under $30"
        cuisine: Type of cuisine (e.g., "Italian", "Japanese").
        location: City or area to search (e.g., "New York").
        price_range: Price level ("$", "$$", "$$$", "$$$$").
        dietary_restrictions: List of dietary requirements.
        limit: Maximum number of results (1-10).

    Returns:
        RestaurantSearchResult: Structured search results from SearchAPI.
    """
    # Check if MCP Gateway is configured
    if not is_mcp_configured():
        logger.warning("MCP Gateway not configured, returning empty result")
        return RestaurantSearchResult(
            query=query,
            total_results=0,
            restaurants=[],
            search_location=location or "",
            search_filters={},
            data_source="searchapi",
            notes="MCP Gateway not configured. Set GATEWAY_URL and Cognito credentials.",
        )

    logger.info(f"Starting SearchAPI restaurant search: '{query}'")

    # Build search parameters - include query for SearchAPI
    search_params = {
        "query": query,
        "cuisine": cuisine or "",
        "location": location or "",
        "price_range": price_range or "$$",
        "dietary_restrictions": dietary_restrictions or [],
        "limit": min(max(1, limit), 10),
    }

    logger.debug(f"Search parameters: {search_params}")

    try:
        # Call the Lambda function via MCP Gateway (Lambda uses SearchAPI)
        response = await call_mcp_tool(SEARCH_RESTAURANTS_TOOL, search_params)

        logger.info(f"SearchAPI tool response received, type: {type(response).__name__}")
        logger.debug(f"SearchAPI tool full response: {json.dumps(response, indent=2) if isinstance(response, dict) else str(response)[:1000]}")

        # Parse response into structured result
        result = parse_search_result(response, query, search_params)

        logger.info(f"Found {result.total_results} restaurants via SearchAPI")

        return result

    except Exception as e:
        logger.error(f"Restaurant data agent failed: {e}")

        return RestaurantSearchResult(
            query=query,
            total_results=0,
            restaurants=[],
            search_location=search_params.get("location", ""),
            search_filters=_convert_to_string_dict(search_params),
            data_source="searchapi",
            notes=f"Search error: {str(e)}",
        )
