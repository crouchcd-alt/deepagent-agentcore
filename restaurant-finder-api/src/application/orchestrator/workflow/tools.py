import json
import uuid
from typing import Annotated, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from loguru import logger

from src.application.orchestrator.workflow.agents.restaurant_explorer_agent import (
    run_restaurant_explorer,
)
from src.application.orchestrator.workflow.agents.restaurant_data_agent import (
    run_restaurant_data_agent,
)
from src.application.orchestrator.workflow.agents.restaurant_research_agent import (
    run_restaurant_research,
)
from src.config import settings
from src.domain.models import RestaurantSearchResult
from src.infrastructure.memory import get_memory_instance


@tool
async def restaurant_explorer_tool(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """
    BACKUP ONLY - Browser-based web search for restaurants.

    WARNING: This tool is SLOW and EXPENSIVE. Only use as a LAST RESORT when:
    1. restaurant_data_tool returned fewer than 4 results, OR
    2. User explicitly asks for "trending", "new", or "latest" restaurants

    DO NOT use this for normal restaurant searches - use restaurant_data_tool instead.

    Args:
        query: Search request with cuisine, location, price, dietary needs.

    Returns:
        JSON with restaurants (name, cuisine, rating, price, address, features).
    """
    # Extract thread_id from config for browser session isolation
    # Generate a unique UUID as fallback to avoid session conflicts
    configurable = config.get("configurable", {}) if config else {}
    thread_id = configurable.get("thread_id") or str(uuid.uuid4())

    result: RestaurantSearchResult = await run_restaurant_explorer(
        query=query,
        thread_id=thread_id,
    )
    return result.model_dump_json(indent=2)


@tool
async def restaurant_data_tool(
    query: str,
    cuisine: str = "",
    location: str = "",
    price_range: str = "$$",
    dietary_restrictions: list[str] = None,
    limit: int = 5,
) -> str:
    """
    PRIMARY TOOL - ALWAYS use this first for ANY restaurant search.

    Fast, reliable restaurant search via Google Local API. Returns structured
    data with ratings, reviews, addresses, phone numbers, hours, and more.

    IMPORTANT: This is your go-to tool for all restaurant searches. Only use
    browser tools (restaurant_explorer_tool) if this returns fewer than 4 results.

    Args:
        query: What restaurants to find (e.g., "best pizza near Times Square").
        cuisine: Cuisine type (Italian, Japanese, etc.).
        location: City or area to search (REQUIRED for best results).
        price_range: "$", "$$", "$$$", or "$$$$".
        dietary_restrictions: List like ["Vegetarian", "Gluten-Free"].
        limit: Max results (1-10).

    Returns:
        JSON with restaurants including ratings, addresses, hours, phone, and more.
    """
    result: RestaurantSearchResult = await run_restaurant_data_agent(
        query=query,
        cuisine=cuisine,
        location=location,
        price_range=price_range,
        dietary_restrictions=dietary_restrictions or [],
        limit=limit,
    )
    return result.model_dump_json(indent=2)


@tool
async def memory_retrieval_tool(
    query: str,
    memory_types: list[Literal["preferences", "facts", "summaries"]],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """
    Retrieve user's stored preferences, facts, or conversation summaries.
    Use for personalization before making recommendations.

    Args:
        query: Search query for semantic matching.
        memory_types: List of types to retrieve:
            - "preferences": Dietary, cuisine, price, location preferences
            - "facts": Details from past conversations
            - "summaries": Current session context

    Returns:
        JSON with memories organized by type.
    """
    configurable = config.get("configurable", {}) if config else {}
    actor_id = configurable.get("actor_id", "user:default")
    session_id = configurable.get("thread_id", "default_session")

    logger.debug(f"Memory retrieval: query='{query}', types={memory_types}, actor={actor_id}")

    try:
        memory = get_memory_instance()
        retrieved = memory.retrieve_specific_memories(
            query=query,
            actor_id=actor_id,
            session_id=session_id,
            memory_types=memory_types,
            top_k=5,
        )

        # Format results for the agent
        formatted_results = {}
        for mem_type, items in retrieved.items():
            formatted_results[mem_type] = [
                item.get("content", str(item)) for item in items
            ]
            logger.debug(f"Retrieved {len(items)} items for '{mem_type}'")

        result_json = json.dumps(formatted_results, indent=2)

        logger.debug(f"Memory retrieval complete: {len(result_json)} chars")
        return result_json

    except Exception as e:
        logger.error(f"Memory retrieval failed: {e}")
        return json.dumps({"error": str(e), "preferences": [], "facts": [], "summaries": []})


@tool
async def restaurant_research_tool(
    restaurant_name: str,
    location: str,
    research_topics: list[str] = None,
    config: Annotated[RunnableConfig, InjectedToolArg] = None,
) -> str:
    """
    FOLLOW-UP ONLY - Deep research on ONE specific restaurant.

    Use ONLY when user asks for more details about a restaurant that was already
    mentioned or recommended. Examples: "Tell me more about X", "What's on the
    menu at X?", "Does X have parking?", "How do I make a reservation at X?"

    DO NOT use this for initial restaurant searches - use restaurant_data_tool instead.

    Args:
        restaurant_name: Restaurant to research (must be a specific restaurant).
        location: City or area.
        research_topics: Optional topics: "menu", "reviews", "reservations",
                        "parking", "events", "contact", "directions".

    Returns:
        JSON with detailed research findings.
    """
    # Extract thread_id from config for browser session isolation
    configurable = config.get("configurable", {}) if config else {}
    thread_id = configurable.get("thread_id") or str(uuid.uuid4())

    logger.debug(f"Restaurant research: name='{restaurant_name}', location='{location}', topics={research_topics}")

    try:
        result = await run_restaurant_research(
            restaurant_name=restaurant_name,
            location=location,
            research_topics=research_topics,
            thread_id=thread_id,
        )

        logger.debug("Restaurant research complete")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Restaurant research failed: {e}")
        return json.dumps({
            "restaurant_name": restaurant_name,
            "location": location,
            "error": str(e),
            "research_summary": f"Unable to research {restaurant_name}. Please try again.",
        })


# Core tools (always available)
_CORE_TOOLS = [
    restaurant_data_tool,       # MCP Gateway to Lambda (SearchAPI web search)
    memory_retrieval_tool,      # On-demand memory retrieval
]

# Browser-based tools (optional)
_BROWSER_TOOLS = [
    restaurant_explorer_tool,   # Browser-based web search for finding restaurants
    restaurant_research_tool,   # Browser-based detailed research on specific restaurant
]


def get_orchestrator_tools(include_browser_tools: bool | None = None) -> list:
    """
    Get the list of tools available to the orchestrator.

    Args:
        include_browser_tools: Override for browser tools inclusion.
                              If None, uses ENABLE_BROWSER_TOOLS from config.

    Returns:
        List of tools for the orchestrator to use.
    """
    use_browser = include_browser_tools if include_browser_tools is not None else settings.ENABLE_BROWSER_TOOLS

    tools = list(_CORE_TOOLS)

    if use_browser:
        tools.extend(_BROWSER_TOOLS)
        logger.info("Browser tools enabled")
    else:
        logger.info("Browser tools disabled")

    return tools

