"""
Restaurant Explorer Agent - Browser-based restaurant search.

This agent uses AWS Bedrock AgentCore Browser tools to search for restaurants
on the web, extracting and structuring the results using an LLM.

Architecture:
- Direct browser tool invocation (not ReAct) for reliable session control
- LLM-based extraction of structured data from raw web content
- Proper thread_id propagation for browser session isolation
"""

import json
import re
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.domain.models import Restaurant, RestaurantSearchResult, PRICE_RANGE_MAP, PriceRange
from src.domain.prompts import RESTAURANT_EXTRACTION_PROMPT
from src.infrastructure.browser import get_browser_tools_by_name, cleanup_browser_sessions
from src.infrastructure.model import get_model, ModelType, extract_text_content


# =============================================================================
# Constants
# =============================================================================

SEARCH_ENGINE_URL = "https://duckduckgo.com"
SEARCH_TIMEOUT_MS = 15000
MAX_TEXT_FOR_EXTRACTION = 8000


# =============================================================================
# Restaurant Parsing
# =============================================================================

def parse_restaurant(data: dict) -> Restaurant:
    """
    Parse a dictionary into a Restaurant model.

    Args:
        data: Dictionary containing restaurant data.

    Returns:
        Restaurant: Parsed restaurant object with defaults for missing fields.
    """
    price_str = data.get("price_range") or "$$"
    price_range = PRICE_RANGE_MAP.get(price_str, PriceRange.MODERATE)

    # Use `or` to handle both missing keys AND None values
    rating_val = data.get("rating")
    review_val = data.get("review_count")

    return Restaurant(
        name=data.get("name") or "Unknown Restaurant",
        cuisine_type=data.get("cuisine_type") or "Various",
        rating=float(rating_val) if rating_val is not None else 0.0,
        review_count=int(review_val) if review_val is not None else 0,
        price_range=price_range,
        address=data.get("address") or "",
        city=data.get("city") or "",
        features=data.get("features") or [],
        dietary_options=data.get("dietary_options") or [],
        operating_hours=data.get("operating_hours") or "",
        reservation_available=bool(data.get("reservation_available")),
    )


def parse_json_results(json_text: str, query: str) -> RestaurantSearchResult:
    """
    Parse JSON text into a RestaurantSearchResult.

    Args:
        json_text: JSON string containing restaurant array.
        query: Original search query.

    Returns:
        RestaurantSearchResult with parsed restaurants or empty result.
    """
    restaurants = []

    try:
        # Extract JSON array from text (handles surrounding text/markdown)
        json_match = re.search(r'\[[\s\S]*\]', json_text)
        if json_match:
            data = json.loads(json_match.group())
            for item in data:
                if isinstance(item, dict):
                    restaurants.append(parse_restaurant(item))
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse JSON results: {e}")

    if not restaurants:
        return RestaurantSearchResult(
            query=query,
            total_results=0,
            restaurants=[],
            search_location="",
            search_filters={},
            data_source="browser",
            notes=f"No structured results extracted. Raw output:\n{json_text[:2000]}",
        )

    return RestaurantSearchResult(
        query=query,
        total_results=len(restaurants),
        restaurants=restaurants,
        search_location="",
        search_filters={},
        data_source="browser",
        notes="Results extracted from web search.",
    )


# =============================================================================
# LLM Extraction
# =============================================================================

async def extract_restaurants_from_text(raw_text: str, query: str) -> str:
    """
    Use an LLM to extract structured restaurant data from raw web content.

    Args:
        raw_text: Raw text extracted from browser.
        query: Original search query for context.

    Returns:
        JSON string containing extracted restaurant data.
    """
    try:
        model = get_model(temperature=0.1, model_type=ModelType.EXTRACTION)

        messages = [
            SystemMessage(content=RESTAURANT_EXTRACTION_PROMPT.prompt),
            HumanMessage(
                content=(
                    f"<search_query>{query}</search_query>\n\n"
                    f"<web_content>\n{raw_text[:MAX_TEXT_FOR_EXTRACTION]}\n</web_content>"
                )
            ),
        ]

        response = await model.ainvoke(messages)
        result = extract_text_content(response.content)

        logger.info(f"LLM extraction completed, response length: {len(result)}")
        return result

    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return "[]"


# =============================================================================
# Browser Operations
# =============================================================================

async def search_web(query: str, config: dict) -> str:
    """
    Perform a web search using browser tools.

    Args:
        query: Search query.
        config: Browser config with thread_id for session isolation.

    Returns:
        Combined raw text from search results page.
    """
    tools = get_browser_tools_by_name()
    results = []

    # Build search URL
    search_query = f"{query} restaurants reviews"
    search_url = f"{SEARCH_ENGINE_URL}/?q={search_query.replace(' ', '+')}&ia=web"

    # Navigate to search engine
    logger.info(f"Navigating to: {search_url}")
    await tools["navigate_browser"].ainvoke({"url": search_url}, config=config)

    # Wait for results to load
    logger.info("Waiting for search results...")
    await tools["wait_for_element"].ainvoke(
        {
            "selector": "[data-testid='result'], .result, .results, article",
            "timeout": SEARCH_TIMEOUT_MS,
            "state": "visible",
        },
        config=config,
    )

    # Extract page content
    logger.info("Extracting page content...")
    page_text = await tools["extract_text"].ainvoke({}, config=config)
    results.append(str(page_text))

    # Extract links for additional context
    logger.info("Extracting hyperlinks...")
    links = await tools["extract_hyperlinks"].ainvoke({}, config=config)
    results.append(f"Links found: {links}")

    return "\n\n".join(results)


# =============================================================================
# Main Entry Point
# =============================================================================

async def run_restaurant_explorer(
    query: str,
    thread_id: str | None = None,
) -> RestaurantSearchResult:
    """
    Search for restaurants using browser automation and LLM extraction.

    This is the main entry point for the restaurant explorer agent. It:
    1. Opens a browser session with DuckDuckGo
    2. Extracts raw text from search results
    3. Uses an LLM to parse the text into structured restaurant data

    Args:
        query: Search query describing restaurants to find.
            Examples:
            - "Italian restaurants in San Francisco"
            - "Vegetarian Thai food under $30"
            - "Fine dining with outdoor seating in NYC"
        thread_id: Browser session identifier. Each conversation should use
            a unique thread_id to avoid session conflicts.

    Returns:
        RestaurantSearchResult: Structured search results.
    """
    effective_thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": effective_thread_id}}

    logger.info(f"Starting restaurant search: '{query}' (thread_id={effective_thread_id})")

    try:
        # Step 1: Search the web
        raw_content = await search_web(query, config)
        logger.info(f"Browser search completed, content length: {len(raw_content)}")

        # Step 2: Extract structured data using LLM
        extracted_json = await extract_restaurants_from_text(raw_content, query)

        # Step 3: Parse into result model
        return parse_json_results(extracted_json, query)

    except Exception as e:
        logger.error(f"Restaurant search failed: {e}")
        return RestaurantSearchResult(
            query=query,
            total_results=0,
            restaurants=[],
            search_location="",
            search_filters={},
            data_source="browser",
            notes=f"Search error: {str(e)}",
        )

    finally:
        # Always cleanup browser session
        try:
            await cleanup_browser_sessions()
            logger.info(f"Browser session cleaned up (thread_id={effective_thread_id})")
        except Exception as cleanup_error:
            logger.warning(f"Browser cleanup failed: {cleanup_error}")
