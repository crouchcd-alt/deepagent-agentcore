"""
Restaurant Research Agent - Detailed research on specific restaurants.

This agent uses AWS Bedrock AgentCore Browser tools to research detailed
information about a specific restaurant, including menu, reviews, hours,
contact info, and other specifics.
"""

import json
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.domain.prompts import RESEARCH_EXTRACTION_PROMPT
from src.infrastructure.browser import get_browser_tools_by_name, cleanup_browser_sessions
from src.infrastructure.model import get_model, ModelType, extract_text_content


# =============================================================================
# Constants
# =============================================================================

SEARCH_ENGINE_URL = "https://duckduckgo.com"
SEARCH_TIMEOUT_MS = 15000
MAX_TEXT_FOR_EXTRACTION = 10000


# =============================================================================
# LLM Extraction
# =============================================================================

async def extract_research_from_text(
    raw_text: str,
    restaurant_name: str,
    location: str,
) -> dict:
    """
    Use an LLM to extract structured research data from raw web content.

    Args:
        raw_text: Raw text extracted from browser.
        restaurant_name: Name of restaurant being researched.
        location: Location of the restaurant.

    Returns:
        Dictionary containing extracted research data.
    """
    try:
        model = get_model(temperature=0.1, model_type=ModelType.EXTRACTION)

        messages = [
            SystemMessage(content=RESEARCH_EXTRACTION_PROMPT.prompt),
            HumanMessage(
                content=(
                    f"<research_target>\n"
                    f"<restaurant_name>{restaurant_name}</restaurant_name>\n"
                    f"<location>{location}</location>\n"
                    f"</research_target>\n\n"
                    f"<web_content>\n{raw_text[:MAX_TEXT_FOR_EXTRACTION]}\n</web_content>"
                )
            ),
        ]

        response = await model.ainvoke(messages)
        result_text = extract_text_content(response.content)

        logger.info(f"LLM research extraction completed, response length: {len(result_text)}")

        # Parse JSON from response
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse research JSON: {e}")

        # Return raw text if JSON parsing fails
        return {
            "restaurant_name": restaurant_name,
            "location": {"city": location},
            "research_summary": result_text[:2000],
            "parse_error": "Could not structure the response"
        }

    except Exception as e:
        logger.error(f"LLM research extraction failed: {e}")
        return {
            "restaurant_name": restaurant_name,
            "error": str(e)
        }


# =============================================================================
# Browser Operations
# =============================================================================

async def search_restaurant_details(
    restaurant_name: str,
    location: str,
    topics: list[str] | None,
    config: dict,
) -> str:
    """
    Perform web searches to gather restaurant details.

    Args:
        restaurant_name: Name of the restaurant.
        location: City/area location.
        topics: Specific topics to research.
        config: Browser config with thread_id.

    Returns:
        Combined raw text from search results.
    """
    tools = get_browser_tools_by_name()
    results = []

    # Build search queries based on topics
    base_query = f"{restaurant_name} {location} restaurant"
    search_queries = [base_query]

    if topics:
        topic_queries = {
            "menu": f"{restaurant_name} {location} menu prices",
            "reviews": f"{restaurant_name} {location} reviews ratings",
            "reservations": f"{restaurant_name} {location} reservations booking",
            "parking": f"{restaurant_name} {location} parking directions",
            "events": f"{restaurant_name} {location} events happy hour live music",
            "contact": f"{restaurant_name} {location} phone address contact",
        }
        for topic in topics:
            if topic.lower() in topic_queries:
                search_queries.append(topic_queries[topic.lower()])
    else:
        # Default: add reviews query
        search_queries.append(f"{restaurant_name} {location} reviews yelp")

    # Execute searches (limit to 2 to avoid too much time)
    for i, query in enumerate(search_queries[:2]):
        try:
            search_url = f"{SEARCH_ENGINE_URL}/?q={query.replace(' ', '+')}&ia=web"

            logger.info(f"Research search {i+1}: {search_url}")
            await tools["navigate_browser"].ainvoke({"url": search_url}, config=config)

            # Wait for results
            await tools["wait_for_element"].ainvoke(
                {
                    "selector": "[data-testid='result'], .result, .results, article, .web-result",
                    "timeout": SEARCH_TIMEOUT_MS,
                    "state": "visible",
                },
                config=config,
            )

            # Extract content
            page_text = await tools["extract_text"].ainvoke({}, config=config)
            results.append(f"=== Search: {query} ===\n{str(page_text)}")

            # Extract links for reference
            links = await tools["extract_hyperlinks"].ainvoke({}, config=config)
            results.append(f"Links: {links}")

        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")
            results.append(f"=== Search failed: {query} ===\nError: {e}")

    return "\n\n".join(results)


# =============================================================================
# Main Entry Point
# =============================================================================

async def run_restaurant_research(
    restaurant_name: str,
    location: str,
    research_topics: list[str] | None = None,
    thread_id: str | None = None,
) -> dict:
    """
    Research detailed information about a specific restaurant.

    This function:
    1. Searches the web for the restaurant
    2. Extracts detailed information using an LLM
    3. Returns structured research findings

    Args:
        restaurant_name: Name of the restaurant to research.
        location: City or area where the restaurant is located.
        research_topics: Optional list of specific topics to focus on.
        thread_id: Browser session identifier.

    Returns:
        Dictionary with detailed research findings.
    """
    effective_thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": effective_thread_id}}

    logger.info(f"Starting restaurant research: '{restaurant_name}' in {location}")
    logger.info(f"Research topics: {research_topics}")

    try:
        # Step 1: Search the web for details
        raw_content = await search_restaurant_details(
            restaurant_name=restaurant_name,
            location=location,
            topics=research_topics,
            config=config,
        )
        logger.info(f"Web research completed, content length: {len(raw_content)}")

        # Step 2: Extract structured data using LLM
        research_data = await extract_research_from_text(
            raw_text=raw_content,
            restaurant_name=restaurant_name,
            location=location,
        )

        # Add metadata
        research_data["research_topics"] = research_topics or ["general"]
        research_data["data_source"] = "web_research"

        return research_data

    except Exception as e:
        logger.error(f"Restaurant research failed: {e}")
        return {
            "restaurant_name": restaurant_name,
            "location": {"city": location},
            "error": str(e),
            "research_summary": f"Unable to complete research for {restaurant_name}.",
        }

    finally:
        # Cleanup browser session
        try:
            await cleanup_browser_sessions()
            logger.info(f"Browser session cleaned up (thread_id={effective_thread_id})")
        except Exception as cleanup_error:
            logger.warning(f"Browser cleanup failed: {cleanup_error}")
